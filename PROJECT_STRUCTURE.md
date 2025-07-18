# Project Structure

## Overview

This project is a cross-platform, GUI-based comic translation tool that leverages state-of-the-art AI models for text detection, OCR, inpainting, and translation. It is designed for extensibility, internationalization, and high performance.

---

## Directory Tree

```
comic-translate-MT/
│
├── app/                # Application logic, UI, controllers, project management
│   ├── controllers/    # Controllers for image, text, project, and rectangle management
│   ├── icon_resource.py
│   ├── projects/       # Project state management and serialization
│   ├── thread_worker.py
│   ├── translations/   # Qt translation files (.ts) and compiled resources
│   └── ui/             # All UI components, widgets, and themes
│
├── modules/            # Core processing modules (detection, OCR, inpainting, translation, rendering, utils)
│   ├── detection/      # Text and bubble detection models and utilities
│   ├── inpainting/     # Inpainting models for text removal
│   ├── ocr/            # OCR engines and factories
│   ├── rendering/      # Text rendering and wrapping
│   ├── translation/    # Translation engines (LLM and traditional)
│   └── utils/          # Shared utilities (file handling, pipeline, textblock, etc.)
│
├── fonts/              # User-imported or bundled fonts
├── models/             # Downloaded or bundled model files (not in repo)
├── pipeline.py         # Main pipeline logic for batch and single image processing
├── controller.py       # Main application controller, connects UI and pipeline
├── comic.py            # Application entry point (main)
├── requirements.txt    # Python dependencies
├── requirements-dev.txt
├── README.md           # Main documentation (multi-language in docs/)
├── docs/               # Additional documentation in multiple languages
└── tests/              # (Empty or for future test code)
```

---

## Directory and File Explanations

### app/
- **controllers/**: MVC-style controllers for different UI and data domains.
  - `image.py`: Manages image loading, state, and navigation.
  - `projects.py`: Handles project save/load, serialization, and UI updates.
  - `rect_item.py`: Manages rectangle (bounding box) UI and logic.
  - `text.py`: Handles text block editing and rendering.
- **icon_resource.py**: Compiled Qt resource file for icons.
- **projects/**: Project state management and serialization.
  - `project_state.py`: Save/load project state to/from .ctpr files (msgpack+images).
  - `parsers.py`: Custom encoders/decoders for project serialization.
- **thread_worker.py**: Threading utilities for running background tasks in the GUI.
- **translations/**: Qt .ts files for UI translation (one per language), plus compiled resources.
  - `ct_*.ts`: Source translation files for each supported language.
  - `ct_translations.py`: Compiled resource loader for translations.
- **ui/**: All UI components, widgets, and themes.
  - `main_window.py`: Main application window and layout.
  - `canvas/`: Image viewer, rectangle, and text item widgets.
  - `commands/`: Undo/redo command pattern for UI actions.
  - `dayu_widgets/`: Custom widgets, themes, and static assets (SVGs, QSS, etc.).
  - `settings/`: Settings UI and logic.
  - `list_view.py`, `messages.py`: Additional UI components.

### modules/
- **detection/**: Text and bubble detection models and utilities.
  - `base.py`: Abstract base class for detection engines.
  - `factory.py`: Factory for selecting detection model.
  - `rtdetr_v2.py`: RT-DETR-V2 model for text/bubble detection.
  - `utils/`: Slicing, box merging, and general detection utilities.
- **ocr/**: Multiple OCR backends and factory.
  - `base.py`: Abstract base class for OCR engines.
  - `factory.py`: Factory for selecting OCR engine.
  - `doctr_ocr.py`, `manga_ocr/`, `pororo/`, `paddle_ocr.py`, `gpt_ocr.py`, etc.: Implementations for each OCR backend.
- **inpainting/**: Text removal using LaMa, AOT, MI-GAN, etc.
  - `base.py`: Abstract base class for inpainting models.
  - `lama.py`, `aot.py`, `mi_gan.py`: Model implementations.
  - `schema.py`: Configuration schemas for inpainting.
- **translation/**: LLM and traditional translation engines.
  - `base.py`: Abstract base class for translation engines.
  - `factory.py`: Factory for selecting translation engine.
  - `deepl.py`, `google.py`, `microsoft.py`, `yandex.py`, `llm/`: Implementations for each translation backend.
- **rendering/**: Text rendering and layout.
  - `render.py`: Main rendering logic.
  - `hyphen_textwrap.py`: Advanced text wrapping.
- **utils/**: Shared utilities.
  - `file_handler.py`: File and archive handling (including extraction).
  - `pipeline_utils.py`: Pipeline helpers (mask generation, language codes, etc.).
  - `textblock.py`: Data structure for text blocks.
  - `download.py`: Model download and verification.
  - `archives.py`: Archive extraction and creation.
  - `translator_utils.py`: Helper functions for translation and image encoding.

### Other Files
- **pipeline.py**: Main pipeline logic for batch and single image processing. Orchestrates detection, OCR, inpainting, translation, and rendering.
- **controller.py**: Main application controller, connects UI and pipeline, manages state and signals.
- **comic.py**: Application entry point. Sets up QApplication, loads translations, and launches the main window.
- **requirements.txt**: Python dependencies for running the application.
- **requirements-dev.txt**: Development dependencies.
- **README.md**: Main documentation (with links to multi-language docs in `docs/`).
- **docs/**: Additional documentation in multiple languages.
- **fonts/**: User-imported or bundled fonts for rendering translated text.
- **models/**: Downloaded or bundled model files (not included in repo, managed at runtime).
- **tests/**: Placeholder for future test code.

---

## Extensibility Points

- **Detection, OCR, Inpainting, Translation**: All use factory patterns and abstract base classes. New models can be added by implementing the base class and registering in the factory.
- **UI**: Modular widgets and commands allow for easy extension of the interface.
- **Internationalization**: Add new Qt .ts files for additional languages.
- **Project State**: Project save/load is versioned and extensible for future features.

---

## Relationships

- The `app/` directory handles all user interaction, UI, and project management.
- The `modules/` directory contains all core AI and processing logic, designed for modularity and easy extension.
- The `pipeline.py` file is the orchestrator, connecting UI actions to the processing modules.
- The `controller.py` file acts as the glue between the UI and the pipeline, managing state and signals.
- The `comic.py` file is the entry point, responsible for application startup, translation loading, and launching the main window. 