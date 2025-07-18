# Project Plan

## 1. Project Goals

- **Universal Comic Translation**: Enable users to translate comics of any format and language using state-of-the-art AI models.
- **User-Friendly GUI**: Provide a modern, intuitive, and responsive interface for both automatic and manual workflows.
- **Extensibility**: Allow easy integration of new detection, OCR, inpainting, and translation models via factory patterns.
- **Internationalization**: Support multiple UI languages and community-contributed translations.
- **Project Management**: Allow users to save, load, and share complete translation projects.
- **Performance**: Optimize for speed, memory usage, and batch processing.

---

## 2. Milestones & Timeline

### MVP (v1.0)
- [x] Image and archive loading (images, CBZ, CBR, PDF, EPUB, etc.)
- [x] Text block detection (RT-DETR-V2, YOLO, etc.)
- [x] OCR (doctr, manga-ocr, Pororo, PaddleOCR, GPT-4 Vision, etc.)
- [x] Inpainting (LaMa, AOT, MI-GAN)
- [x] Translation (LLM and traditional engines)
- [x] Text rendering and export
- [x] Project save/load (.ctpr)
- [x] Basic internationalization (Qt .ts files)

### v1.1 - v1.5
- [x] Manual correction mode (edit boxes, OCR, translations, inpainting)
- [x] Undo/redo for all actions
- [x] OCR and translation caching (see PR #303)
- [x] Improved batch processing and error handling
- [x] Enhanced UI/UX (themes, accessibility, drag-and-drop)
- [x] Community translation support

### v2.0+
- [ ] Plugin system for new models and UI features
- [ ] Cloud-based processing for heavy models
- [ ] Collaborative project features
- [ ] Advanced text layout and font matching
- [ ] Mobile/web versions
- [ ] Automated and manual testing suite

---

## 3. Technical Roadmap

### Core Architecture
- **Factory Patterns**: All detection, OCR, inpainting, and translation modules use abstract base classes and factories for easy extension.
- **Pipeline Orchestration**: `ComicTranslatePipeline` coordinates all processing steps and manages state.
- **Project Serialization**: Projects are saved as `.ctpr` (zip+msgpack+images+patches) for portability and robustness.
- **UI Modularity**: Widgets, commands, and settings are modular for future UI features.

### Extensibility Strategy
- **Adding New Models**: Implement the base class (DetectionEngine, OCREngine, InpaintModel, TranslationEngine) and register in the appropriate factory.
- **UI Extensions**: Add new widgets or commands in `app/ui/` and connect via controllers.
- **Internationalization**: Add new `.ts` files in `app/translations/` and update resource loader.
- **Project State**: Extend project serialization for new features (e.g., collaborative editing, cloud sync).

### Testing & Documentation
- **Testing**: (Planned) Automated tests for pipeline, UI, and model integration.
- **Documentation**: Maintain detailed docs for structure, workflow, extensibility, and user guides.
- **Community Contributions**: Guidelines for contributing code, translations, and documentation.

---

## 4. Community & Contribution
- **Translation**: Community can contribute new `.ts` files for additional languages.
- **Plugins**: (Planned) Plugin API for new models and UI features.
- **Issue Tracking**: Use GitHub Issues for bugs, feature requests, and discussion.
- **Pull Requests**: Encourage PRs for bugfixes, new models, and documentation improvements.

---

## 5. Future Directions
- **More Backends**: Add support for additional OCR, translation, and inpainting models.
- **Cloud Processing**: Offload heavy computation to cloud services for users without powerful hardware.
- **Collaboration**: Real-time collaborative editing and translation.
- **Mobile/Web**: Port to mobile and web platforms.
- **Advanced Rendering**: Improved text layout, font matching, and style transfer.

---

## 6. Timeline (Example)

| Milestone         | Target Date   | Status  |
|-------------------|--------------|---------|
| MVP Release       | 2024-07-01   | Done    |
| Manual Mode       | 2024-08-01   | Done    |
| Caching           | 2024-08-15   | Done    |
| UI/UX Enhancements| 2024-09-01   | Done    |
| Plugin System     | 2024-12-01   | Planned |
| Cloud Processing  | 2025-03-01   | Planned |
| Collaboration     | 2025-06-01   | Planned |
| Mobile/Web        | 2025-09-01   | Planned |

---

## 7. Summary

This project aims to be the most flexible, powerful, and user-friendly comic translation platform available. Its modular design, extensibility, and community focus ensure it will continue to evolve and support new technologies and user needs. 