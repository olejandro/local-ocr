# Next Steps (in the order of priority)

1. Environment verification:
	- Install/verify `pandas`, `pypdf`, `easyocr`, `transformers`, `torch`, `pillow`, `numpy`
	- Run one baseline PDF end-to-end and verify output workbook
2. CLI validation + logging:
	- Validate model/PDF paths and threshold ranges
	- Add structured logs (page/image/table context + timings)
3. Accuracy tuning:
	- Tune `table_threshold` and `cell_threshold` on a validation PDF set
	- Improve OCR preprocessing (grayscale/contrast/deskew)
	- Improve column anchoring for merged/missing cells
4. Testing + metrics:
	- Add smoke tests (no-image pages, corrupt images, empty OCR)
	- Track runtime and extraction quality metrics per run
