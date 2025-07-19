import os
import json
import threading
import hashlib
import cv2
import shutil
import numpy as np
from datetime import datetime
from typing import List
from PySide6 import QtCore
from PySide6.QtGui import QColor

from modules.detection.processor import TextBlockDetector
from modules.ocr.processor import OCRProcessor
from modules.translation.processor import Translator
from modules.utils.textblock import TextBlock, sort_blk_list
from modules.utils.pipeline_utils import inpaint_map, get_config
from modules.rendering.render import get_best_render_area, pyside_word_wrap
from modules.utils.pipeline_utils import generate_mask, get_language_code, is_directory_empty
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations, set_upper_case
from modules.utils.archives import make

from modules.utils.cache import PersistentCache, ThreadSafeLRUCache, get_cache_manager
from modules.utils.exceptions import handle_exception_chain, CacheError, ComicTranslateException, OCRError
from modules.utils.logging_config import get_pipeline_logger

from app.ui.canvas.rectangle import MoveableRectItem
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.canvas.save_renderer import ImageSaveRenderer

class ComicTranslatePipeline:
    def __init__(self, main_page):
        self.main_page = main_page
        self.block_detector_cache = None
        self.inpainter_cache = None
        self.cached_inpainter_key = None
        self.ocr = OCRProcessor()

        # Setup persistent OCR cache
        cache_dir = os.path.join(os.getcwd(), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        ocr_cache_file = os.path.join(cache_dir, "ocr_cache.msgpack")
        try:
            self.ocr_cache = PersistentCache(cache_file=ocr_cache_file, max_size=500, auto_save=True)
        except Exception as e:
            # Fallback to in-memory LRU cache if persistent fails
            self.ocr_cache = ThreadSafeLRUCache(max_size=500)
        self.cache_lock = threading.RLock()

        # Logger for pipeline
        self.logger = get_pipeline_logger()

    def clear_ocr_cache(self):
        """Clear the OCR cache."""
        try:
            with self.cache_lock:
                self.ocr_cache.clear()
            self.logger.info("OCR cache cleared", component="Pipeline")
        except CacheError as e:
            self.logger.warning(f"Failed to clear OCR cache: {e}", component="Pipeline")

    def load_box_coords(self, blk_list: List[TextBlock]):
        self.main_page.image_viewer.clear_rectangles()
        if self.main_page.image_viewer.hasPhoto() and blk_list:
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                rect = QtCore.QRectF(0, 0, x2 - x1, y2 - y1)
                rect_item = MoveableRectItem(rect, self.main_page.image_viewer.photo)
                if blk.tr_origin_point:
                    rect_item.setTransformOriginPoint(QtCore.QPointF(*blk.tr_origin_point))
                rect_item.setPos(x1, y1)
                rect_item.setRotation(blk.angle)
                self.main_page.connect_rect_item_signals(rect_item)
                self.main_page.image_viewer.rectangles.append(rect_item)

            rect = self.main_page.rect_item_ctrl.find_corresponding_rect(self.main_page.blk_list[0], 0.5)
            self.main_page.image_viewer.select_rectangle(rect)
            self.main_page.set_tool('box')

    def detect_blocks(self, load_rects=True):
        if self.main_page.image_viewer.hasPhoto():
            if self.block_detector_cache is None:
                self.block_detector_cache = TextBlockDetector(self.main_page.settings_page)
            image = self.main_page.image_viewer.get_cv2_image()
            try:
                blk_list = self.block_detector_cache.detect(image)
                return blk_list, load_rects
            except Exception as e:
                exc = handle_exception_chain(e)
                self.logger.error("Block detection failed", exception=exc, component="Pipeline")
                raise exc

    def on_blk_detect_complete(self, result):
        blk_list, load_rects = result
        source_lang = self.main_page.s_combo.currentText()
        source_lang_en = self.main_page.lang_mapping.get(source_lang, source_lang)
        rtl = source_lang_en == 'Japanese'
        blk_list = sort_blk_list(blk_list, rtl)
        self.main_page.blk_list = blk_list
        if load_rects:
            self.load_box_coords(blk_list)

    def manual_inpaint(self):
        image_viewer = self.main_page.image_viewer
        settings_page = self.main_page.settings_page
        mask = image_viewer.get_mask_for_inpainting()
        image = image_viewer.get_cv2_image()

        if self.inpainter_cache is None or self.cached_inpainter_key != settings_page.get_tool_selection('inpainter'):
            device = 'cuda' if settings_page.is_gpu_enabled() else 'cpu'
            inpainter_key = settings_page.get_tool_selection('inpainter')
            InpainterClass = inpaint_map[inpainter_key]
            self.inpainter_cache = InpainterClass(device)
            self.cached_inpainter_key = inpainter_key

        config = get_config(settings_page)
        try:
            inpaint_input_img = self.inpainter_cache(image, mask, config)
            inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
            return inpaint_input_img
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.error("Inpainting failed", exception=exc, component="Pipeline")
            raise exc

    def inpaint_complete(self, patch_list):
        self.main_page.apply_inpaint_patches(patch_list)
        self.main_page.image_viewer.clear_brush_strokes()
        self.main_page.undo_group.activeStack().endMacro()

    def get_inpainted_patches(self, mask: np.ndarray, inpainted_image: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patches = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            patch = inpainted_image[y:y+h, x:x+w]
            patches.append({'bbox': (x, y, w, h), 'cv2_img': patch.copy()})
        return patches

    def inpaint(self):
        mask = self.main_page.image_viewer.get_mask_for_inpainting()
        painted = self.manual_inpaint()
        return self.get_inpainted_patches(mask, painted)

    def get_selected_block(self):
        rect = self.main_page.image_viewer.selected_rect
        srect = rect.mapRectToScene(rect.rect())
        blk = self.main_page.rect_item_ctrl.find_corresponding_text_block(srect.getCoords())
        return blk

    def _generate_image_hash(self, image):
        """Generate a robust hash for the image to use as cache key."""
        try:
            # Sample image data and include shape/dtype
            sample = image[::10, ::10].tobytes()
            meta = f"{image.shape}-{image.dtype}".encode()
            return hashlib.sha256(sample + meta).hexdigest()
        except Exception:
            # Fallback to shape-only hash
            data = f"{getattr(image, 'shape', '')}-{getattr(image, 'dtype', '')}".encode()
            return hashlib.sha256(data).hexdigest()

    def _get_cache_key(self, image, source_lang):
        image_hash = self._generate_image_hash(image)
        ocr_model = self.main_page.settings_page.get_tool_selection('ocr')
        return f"{image_hash}:{ocr_model}:{source_lang}"

    def _is_ocr_cached(self, cache_key):
        with self.cache_lock:
            return cache_key in self.ocr_cache

    def _cache_ocr_results(self, cache_key, blk_list):
        try:
            results = {self._get_block_id(blk): getattr(blk, 'text', '') or '' for blk in blk_list}
            with self.cache_lock:
                self.ocr_cache.put(cache_key, results) if hasattr(self.ocr_cache, 'put') else self.ocr_cache.__setitem__(cache_key, results)
            self.logger.debug("OCR results cached", cache_key=cache_key, component="Pipeline")
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.warning("Failed to cache OCR results", exception=exc, component="Pipeline")

    def _get_cached_text_for_block(self, cache_key, block):
        try:
            with self.cache_lock:
                data = self.ocr_cache.get(cache_key, {})
            return data.get(self._get_block_id(block), "")
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.warning("Failed to retrieve cached OCR text", exception=exc, component="Pipeline")
            return ""

    def _get_block_id(self, block):
        try:
            x1, y1, x2, y2 = block.xyxy
            return f"{x1}_{y1}_{x2}_{y2}"
        except Exception:
            return str(id(block))

    def OCR_image(self, single_block=False):
        source_lang = self.main_page.s_combo.currentText()
        if not (self.main_page.image_viewer.hasPhoto() and self.main_page.image_viewer.rectangles):
            return
        image = self.main_page.image_viewer.get_cv2_image()
        cache_key = self._get_cache_key(image, source_lang)

        try:
            if single_block:
                blk = self.get_selected_block()
                if blk is None:
                    return
                if self._is_ocr_cached(cache_key):
                    blk.text = self._get_cached_text_for_block(cache_key, blk)
                    self.logger.info("Using cached OCR for block", block_id=self._get_block_id(blk), component="Pipeline")
                else:
                    self.logger.info("Running OCR on full page", component="Pipeline")
                    self.ocr.initialize(self.main_page, source_lang)
                    blocks = list(self.main_page.blk_list)
                    self.ocr.process(image, blocks)
                    self._cache_ocr_results(cache_key, blocks)
                    blk.text = self._get_cached_text_for_block(cache_key, blk)
            else:
                self.logger.info("Running OCR on full page", component="Pipeline")
                self.ocr.initialize(self.main_page, source_lang)
                blocks = list(self.main_page.blk_list)
                self.ocr.process(image, blocks)
                self._cache_ocr_results(cache_key, blocks)
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.error("OCR processing failed", exception=exc, component="Pipeline")
            raise OCRError(str(exc), cause=exc)

    def translate_image(self, single_block=False):
        source_lang = self.main_page.s_combo.currentText()
        target_lang = self.main_page.t_combo.currentText()
        if not (self.main_page.image_viewer.hasPhoto() and self.main_page.blk_list):
            return
        settings_page = self.main_page.settings_page
        image = self.main_page.image_viewer.get_cv2_image()
        extra_context = settings_page.get_llm_settings().get('extra_context', {})

        translator = Translator(self.main_page, source_lang, target_lang)
        try:
            if single_block:
                blk = self.get_selected_block()
                if blk:
                    translator.translate([blk], image, extra_context)
                    set_upper_case([blk], settings_page.ui.uppercase_checkbox.isChecked())
            else:
                translator.translate(self.main_page.blk_list, image, extra_context)
                set_upper_case(self.main_page.blk_list, settings_page.ui.uppercase_checkbox.isChecked())
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.error("Translation failed", exception=exc, component="Pipeline")
            raise exc

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        try:
            path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_images", archive_bname)
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(os.path.join(path, f"{base_name}_translated{extension}"), image)
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.warning("Failed to skip-save image", exception=exc, component="Pipeline")

    def log_skipped_image(self, directory, timestamp, image_path):
        try:
            os.makedirs(os.path.join(directory, f"comic_translate_{timestamp}"), exist_ok=True)
            with open(os.path.join(directory, f"comic_translate_{timestamp}", "skipped_images.txt"), 'a', encoding='UTF-8') as file:
                file.write(image_path + "\n")
        except Exception as e:
            exc = handle_exception_chain(e)
            self.logger.warning("Failed to log skipped image", exception=exc, component="Pipeline")

    def batch_process(self, selected_paths: List[str] = None):
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total = len(image_list)
        self.logger.info("Batch processing started", total_images=total, component="Pipeline")

        # Monitor memory pressure
        mem_monitor = get_cache_manager()._memory_monitor
        if mem_monitor.is_memory_pressure():
            self.logger.warning("High memory pressure detected, clearing caches", component="Pipeline")
            with self.cache_lock:
                self.ocr_cache.clear()

        for idx, image_path in enumerate(image_list):
            try:
                self.logger.info("Processing image", index=idx, image_path=image_path, component="Pipeline")
                # Emit progress: pre-processing
                self.main_page.progress_update.emit(idx, total, 0, 10, True)

                # Load image safely
                image = cv2.imread(image_path)
                if image is None:
                    raise ComicTranslateException(f"Failed to read image: {image_path}")

                # Skip if marked
                state = self.main_page.image_states.get(image_path, {})
                if state.get('skip', False):
                    self.skip_save_dir = state.get('skip_dir')  # optional
                    self.skip_save(os.path.dirname(image_path), timestamp,
                                   os.path.splitext(os.path.basename(image_path))[0],
                                   os.path.splitext(image_path)[1],
                                   "", image)
                    self.log_skipped_image(os.path.dirname(image_path), timestamp, image_path)
                    continue

                # Detection
                self.main_page.progress_update.emit(idx, total, 1, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    self.logger.info("Batch canceled during detection", component="Pipeline")
                    break
                if self.block_detector_cache is None:
                    self.block_detector_cache = TextBlockDetector(self.main_page.settings_page)
                blk_list = self.block_detector_cache.detect(image)
                if not blk_list:
                    raise ComicTranslateException("No text blocks found")

                # OCR
                self.main_page.progress_update.emit(idx, total, 2, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    self.logger.info("Batch canceled during OCR", component="Pipeline")
                    break
                self.ocr.initialize(self.main_page, self.main_page.image_states[image_path]['source_lang'])
                self.ocr.process(image, blk_list)

                # Sort blocks
                src_en = self.main_page.lang_mapping.get(self.main_page.image_states[image_path]['source_lang'])
                rtl = src_en == 'Japanese'
                blk_list = sort_blk_list(blk_list, rtl)

                # Cache OCR results
                self._cache_ocr_results(self._get_cache_key(image, self.main_page.image_states[image_path]['source_lang']), blk_list)

                # Inpainting & cleaning
                self.main_page.progress_update.emit(idx, total, 3, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    self.logger.info("Batch canceled during inpainting", component="Pipeline")
                    break
                settings = self.main_page.settings_page
                device = 'cuda' if settings.is_gpu_enabled() else 'cpu'
                key = settings.get_tool_selection('inpainter')
                if self.inpainter_cache is None or self.cached_inpainter_key != key:
                    self.inpainter_cache = inpaint_map[key](device)
                    self.cached_inpainter_key = key
                mask = generate_mask(image, blk_list)
                cleaned = self.inpainter_cache(image, mask, get_config(settings))
                cleaned = cv2.convertScaleAbs(cleaned)
                patches = self.get_inpainted_patches(mask, cleaned)
                self.main_page.patches_processed.emit(idx, patches, image_path)
                cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
                if settings.get_export_settings().get('export_inpainted_image'):
                    out_dir = os.path.join(os.path.dirname(image_path), f"comic_translate_{timestamp}", "cleaned_images")
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_cleaned{os.path.splitext(image_path)[1]}"), cleaned_rgb)

                # Translation
                self.main_page.progress_update.emit(idx, total, 4, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    self.logger.info("Batch canceled during translation", component="Pipeline")
                    break
                translator = Translator(self.main_page,
                                        self.main_page.image_states[image_path]['source_lang'],
                                        self.main_page.image_states[image_path]['target_lang'])
                translator.translate(blk_list, image, settings.get_llm_settings().get('extra_context', {}))

                # Export raw and translated texts
                raw = get_raw_text(blk_list)
                trans = get_raw_translation(blk_list)
                try:
                    if settings.get_export_settings().get('export_raw_text'):
                        raw_dir = os.path.join(os.path.dirname(image_path), f"comic_translate_{timestamp}", "raw_texts")
                        os.makedirs(raw_dir, exist_ok=True)
                        with open(os.path.join(raw_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_raw.txt"), 'w', encoding='utf-8') as f:
                            f.write(raw)
                    if settings.get_export_settings().get('export_translated_text'):
                        txt_dir = os.path.join(os.path.dirname(image_path), f"comic_translate_{timestamp}", "translated_texts")
                        os.makedirs(txt_dir, exist_ok=True)
                        with open(os.path.join(txt_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_translated.txt"), 'w', encoding='utf-8') as f:
                            f.write(trans)
                except Exception as e:
                    exc = handle_exception_chain(e)
                    self.logger.warning("Failed to save text files", exception=exc, component="Pipeline")

                # Rendering and final save
                self.main_page.progress_update.emit(idx, total, 5, 10, False)
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    self.logger.info("Batch canceled during rendering", component="Pipeline")
                    break
                render_settings = self.main_page.render_settings()
                format_translations(blk_list, get_language_code(self.main_page.lang_mapping.get(self.main_page.image_states[image_path]['target_lang'])), upper_case=render_settings.upper_case)
                get_best_render_area(blk_list, image, cleaned)
                im_bgr = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)
                save_dir = os.path.join(os.path.dirname(image_path), f"comic_translate_{timestamp}", "translated_images")
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_translated{os.path.splitext(image_path)[1]}")
                renderer = ImageSaveRenderer(im_bgr)
                renderer.apply_patches(self.main_page.image_patches.get(image_path, []))
                renderer.add_state_to_image(self.main_page.image_states[image_path]['viewer_state'])
                try:
                    renderer.save_image(output_path)
                except Exception as e:
                    exc = handle_exception_chain(e)
                    self.logger.warning("Failed to save rendered image", exception=exc, component="Pipeline")

                self.main_page.progress_update.emit(idx, total, 10, 10, False)

            except Exception as e:
                exc = handle_exception_chain(e)
                self.logger.error("Error processing image in batch", exception=exc, component="Pipeline", image_path=image_path)
                self.main_page.image_skipped.emit(image_path, getattr(exc, 'error_code', 'UNKNOWN'), str(exc))
                self.log_skipped_image(os.path.dirname(image_path), timestamp, image_path)
                continue

        # Handle archives post-processing
        archives = self.main_page.file_handler.archive_info
        if archives:
            save_as = self.main_page.settings_page.get_export_settings().get('save_as', {})
            for ai, archive in enumerate(archives):
                if self.main_page.current_worker and self.main_page.current_worker.is_cancelled:
                    self.logger.info("Batch canceled during archive handling", component="Pipeline")
                    break
                ap = archive['archive_path']
                ext = os.path.splitext(ap)[1]
                out_ext = f".{save_as.get(ext.lower(), ext.lower().strip('.'))}"
                in_dir = os.path.join(os.path.dirname(ap), f"comic_translate_{timestamp}", "translated_images", os.path.splitext(os.path.basename(ap))[0])
                try:
                    make(save_as_ext=out_ext, input_dir=in_dir, output_dir=os.path.dirname(ap), output_base_name=os.path.splitext(os.path.basename(ap))[0])
                except Exception as e:
                    exc = handle_exception_chain(e)
                    self.logger.warning("Failed to create archive", exception=exc, component="Pipeline", archive=ap)
                finally:
                    if os.path.exists(in_dir):
                        shutil.rmtree(in_dir)
                    temp_dir = os.path.join(os.path.dirname(ap), f"comic_translate_{timestamp}")
                    if is_directory_empty(temp_dir):
                        shutil.rmtree(temp_dir)

        self.logger.info("Batch processing completed", component="Pipeline")