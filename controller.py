import os
import numpy as np
import shutil
import tempfile
import threading
from typing import Callable, Tuple

from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtCore import QCoreApplication, QThreadPool
from PySide6.QtGui import QUndoGroup, QUndoStack

from collections import deque, OrderedDict

from app.ui.dayu_widgets.qt import MPixmap
from app.ui.main_window import ComicTranslateUI
from app.ui.messages import Messages
from app.thread_worker import GenericWorker

from app.ui.canvas.text_item import TextBlockItem
from app.ui.commands.box import DeleteBoxesCommand

from modules.utils.textblock import TextBlock
from modules.utils.file_handler import FileHandler
from modules.utils.pipeline_utils import validate_settings, validate_ocr, \
                                         validate_translator
from modules.utils.download import get_models, mandatory_models
from modules.detection.utils.general import get_inpaint_bboxes
from modules.utils.translator_utils import is_there_text
from modules.rendering.render import pyside_word_wrap
from modules.utils.pipeline_utils import get_language_code
from modules.utils.translator_utils import format_translations
from pipeline import ComicTranslatePipeline

from app.controllers.image import ImageStateController
from app.controllers.rect_item import RectItemController
from app.controllers.projects import ProjectController
from app.controllers.text import TextController

from modules.utils.exceptions import handle_exception_chain, log_exception
from modules.utils.logging_config import get_ui_logger

for model in mandatory_models:
    get_models(model)

class ComicTranslate(ComicTranslateUI):
    image_processed = QtCore.Signal(int, object, str)
    patches_processed = QtCore.Signal(int, list, str)
    progress_update = QtCore.Signal(int, int, int, int, bool)
    image_skipped = QtCore.Signal(str, str, str)
    blk_rendered = QtCore.Signal(str, int, object)

    def __init__(self, parent=None):
        super(ComicTranslate, self).__init__(parent)

        # thread safety for queue and memory
        self._queue_lock = threading.RLock()

        self.blk_list: list[TextBlock] = []
        self.curr_tblock: TextBlock = None
        self.curr_tblock_item: TextBlockItem = None

        self.image_files = []
        self.selected_batch = []
        self.curr_img_idx = -1
        self.image_states = {}
        self.image_data = {}  # Store the latest version of each image
        self.image_history = OrderedDict()  # Store file path history for all images
        self.in_memory_history = OrderedDict()  # Store cv2 image history for recent images
        self.current_history_index = {}  # Current position in the history for each image
        self.displayed_images = set()  # Set to track displayed images
        self.image_patches = {}
        self.in_memory_patches = OrderedDict()  # Store patches in memory for each image
        self.image_cards = []
        self.current_card = None
        self.max_images_in_memory = 10
        self.loaded_images = []

        self.undo_group = QUndoGroup(self)
        self.undo_stacks: dict[str, QUndoStack] = {}
        self.project_file = None
        self.temp_dir = tempfile.mkdtemp()

        self.pipeline = ComicTranslatePipeline(self)
        self.file_handler = FileHandler()
        self.threadpool = QThreadPool()
        self.current_worker = None

        self.image_ctrl = ImageStateController(self)
        self.rect_item_ctrl = RectItemController(self)
        self.project_ctrl = ProjectController(self)
        self.text_ctrl = TextController(self)

        # connect signals
        self.image_skipped.connect(self.image_ctrl.on_image_skipped)
        self.image_processed.connect(self.image_ctrl.on_image_processed)
        self.image_processed.connect(self._on_image_processed)
        self.patches_processed.connect(self.image_ctrl.on_inpaint_patches_processed)
        self.progress_update.connect(self.update_progress)
        self.blk_rendered.connect(self.text_ctrl.on_blk_rendered)

        self.connect_ui_elements()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.project_ctrl.load_main_page_settings()
        self.settings_page.load_settings()

        self.operation_queue = deque()
        self.is_processing_queue = False

    def manage_memory(self):
        """Evict oldest images and patches when exceeding max_images_in_memory."""
        with self._queue_lock:
            while len(self.in_memory_history) > self.max_images_in_memory:
                old_path, _ = self.in_memory_history.popitem(last=False)
                self.image_data.pop(old_path, None)
                self.in_memory_patches.pop(old_path, None)

    def _on_image_processed(self, index: int, image: object, file_path: str):
        """Store image in history and enforce LRU eviction."""
        try:
            with self._queue_lock:
                self.in_memory_history[file_path] = image
                self.in_memory_history.move_to_end(file_path)
            self.manage_memory()
        except Exception as e:
            exc = handle_exception_chain(e)
            log_exception(exc, logger=get_ui_logger())

    def connect_ui_elements(self):
        # Browsers
        self.image_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.document_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.archive_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.comic_browser_button.sig_files_changed.connect(self.image_ctrl.thread_load_images)
        self.project_browser_button.sig_file_changed.connect(self.project_ctrl.thread_load_project)

        self.save_browser.sig_file_changed.connect(self.image_ctrl.save_current_image)
        self.save_all_browser.sig_file_changed.connect(self.project_ctrl.save_and_make)
        self.save_project_button.clicked.connect(self.project_ctrl.thread_save_project)
        self.save_as_project_button.clicked.connect(self.project_ctrl.thread_save_as_project)

        self.drag_browser.sig_files_changed.connect(self.image_ctrl.thread_load_images)

        self.manual_radio.clicked.connect(self.manual_mode_selected)
        self.automatic_radio.clicked.connect(self.batch_mode_selected)

        # Connect buttons from button_groups
        self.hbutton_group.get_button_group().buttons()[0].clicked.connect(lambda: self.block_detect())
        self.hbutton_group.get_button_group().buttons()[1].clicked.connect(self.ocr)
        self.hbutton_group.get_button_group().buttons()[2].clicked.connect(self.translate_image)
        self.hbutton_group.get_button_group().buttons()[3].clicked.connect(self.load_segmentation_points)
        self.hbutton_group.get_button_group().buttons()[4].clicked.connect(self.inpaint_and_set)
        self.hbutton_group.get_button_group().buttons()[5].clicked.connect(self.text_ctrl.render_text)

        self.undo_tool_group.get_button_group().buttons()[0].clicked.connect(self.undo_group.undo)
        self.undo_tool_group.get_button_group().buttons()[1].clicked.connect(self.undo_group.redo)

        # Connect other buttons and widgets
        self.translate_button.clicked.connect(self.start_batch_process)
        self.cancel_button.clicked.connect(self.cancel_current_task)
        self.set_all_button.clicked.connect(self.text_ctrl.set_src_trg_all)
        self.clear_rectangles_button.clicked.connect(self.image_viewer.clear_rectangles)
        self.clear_brush_strokes_button.clicked.connect(self.image_viewer.clear_brush_strokes)
        self.draw_blklist_blks.clicked.connect(lambda: self.pipeline.load_box_coords(self.blk_list))
        self.change_all_blocks_size_dec.clicked.connect(lambda: self.text_ctrl.change_all_blocks_size(-int(self.change_all_blocks_size_diff.text())))
        self.change_all_blocks_size_inc.clicked.connect(lambda: self.text_ctrl.change_all_blocks_size(int(self.change_all_blocks_size_diff.text())))
        self.delete_button.clicked.connect(self.delete_selected_box)

        # Connect text edit widgets
        self.s_text_edit.textChanged.connect(self.text_ctrl.update_text_block)
        self.t_text_edit.textChanged.connect(self.text_ctrl.update_text_block_from_edit)

        self.s_combo.currentTextChanged.connect(self.text_ctrl.save_src_trg)
        self.t_combo.currentTextChanged.connect(self.text_ctrl.save_src_trg)

        # Connect image viewer signals
        self.image_viewer.rectangle_selected.connect(self.rect_item_ctrl.handle_rectangle_selection)
        self.image_viewer.rectangle_created.connect(self.rect_item_ctrl.handle_rectangle_creation)
        self.image_viewer.rectangle_deleted.connect(self.rect_item_ctrl.handle_rectangle_deletion)
        self.image_viewer.command_emitted.connect(self.push_command)
        self.image_viewer.connect_rect_item.connect(self.rect_item_ctrl.connect_rect_item_signals)
        self.image_viewer.connect_text_item.connect(self.text_ctrl.connect_text_item_signals)

        # Rendering
        self.font_dropdown.currentTextChanged.connect(self.text_ctrl.on_font_dropdown_change)
        self.font_size_dropdown.currentTextChanged.connect(self.text_ctrl.on_font_size_change)
        self.line_spacing_dropdown.currentTextChanged.connect(self.text_ctrl.on_line_spacing_change)
        self.block_font_color_button.clicked.connect(self.text_ctrl.on_font_color_change)
        self.alignment_tool_group.get_button_group().buttons()[0].clicked.connect(self.text_ctrl.left_align)
        self.alignment_tool_group.get_button_group().buttons()[1].clicked.connect(self.text_ctrl.center_align)
        self.alignment_tool_group.get_button_group().buttons()[2].clicked.connect(self.text_ctrl.right_align)
        self.bold_button.clicked.connect(self.text_ctrl.bold)
        self.italic_button.clicked.connect(self.text_ctrl.italic)
        self.underline_button.clicked.connect(self.text_ctrl.underline)
        self.outline_font_color_button.clicked.connect(self.text_ctrl.on_outline_color_change)
        self.outline_width_dropdown.currentTextChanged.connect(self.text_ctrl.on_outline_width_change)
        self.outline_checkbox.stateChanged.connect(self.text_ctrl.toggle_outline_settings)

        # Page List
        self.page_list.currentItemChanged.connect(self.image_ctrl.on_card_selected)
        self.page_list.del_img.connect(self.image_ctrl.handle_image_deletion)
        self.page_list.insert_browser.sig_files_changed.connect(self.image_ctrl.thread_insert)
        self.page_list.toggle_skip_img.connect(self.image_ctrl.handle_toggle_skip_images)
        self.page_list.translate_imgs.connect(self.batch_translate_selected)

    # ... (existing methods unchanged) ...

    def default_error_handler(self, error_tuple: Tuple):
        exctype, value, traceback_str = error_tuple
        exception = handle_exception_chain(value)
        ui_logger = get_ui_logger()
        log_exception(exception, logger=ui_logger)
        QtWidgets.QMessageBox.critical(self, "Error", exception.user_message)
        self.loading.setVisible(False)
        self.enable_hbutton_group()

    def _queue_operation(self, callback: Callable, result_callback: Callable=None, 
                        error_callback: Callable=None, finished_callback: Callable=None, 
                        *args, **kwargs):
        """Queue an operation for sequential execution"""
        with self._queue_lock:
            operation = {
                'callback': callback,
                'result_callback': result_callback,
                'error_callback': error_callback,
                'finished_callback': finished_callback,
                'args': args,
                'kwargs': kwargs,
            }
            self.operation_queue.append(operation)
            if not self.is_processing_queue:
                self._process_next_operation()

    def clear_operation_queue(self):
        """Clear all pending operations in the queue"""
        with self._queue_lock:
            self.operation_queue.clear()

    def cancel_current_task(self):
        """Enhanced cancel that also clears the queue and cleans up resources"""
        if self.current_worker:
            self.current_worker.cancel()
        self.clear_operation_queue()
        self.is_processing_queue = False
        # Attempt GPU cache cleanup if available
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def closeEvent(self, event):
        ui_logger = get_ui_logger()
        try:
            self.settings_page.save_settings()
            self.project_ctrl.save_main_page_settings()
        except Exception as e:
            log_exception(handle_exception_chain(e), logger=ui_logger)
        # Delete temp archive folders
        for archive in self.file_handler.archive_info:
            temp_dir = archive.get('temp_dir')
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                log_exception(handle_exception_chain(e), logger=ui_logger)
        # Delete our own temp dir
        try:
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)
        except Exception as e:
            log_exception(handle_exception_chain(e), logger=ui_logger)
        # Cleanup GPU
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        super().closeEvent(event)