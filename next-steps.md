# Next Steps: Robustness and Accuracy

## 1) Immediate Setup (run first)

1. Install and verify dependencies in the active environment:
	- `pandas`
	- `pypdf`
	- `easyocr`
	- `transformers`
	- `torch`
	- `pillow`
	- `numpy`
2. Run one known PDF end-to-end with CLI flags and confirm:
	- At least one table is detected.
	- DataFrames print without exceptions.
	- Header promotion behavior is acceptable.

## 2) Robustness Hardening (highest priority)

1. Add structured logging (page index, image index, table index, timings, exception context).
2. Add validation for CLI inputs:
	- Model dirs exist.
	- PDF path exists and is a file.
	- Thresholds are in `[0, 1]`.
3. Return richer table metadata (page/image/table IDs, table box, cell count, confidence summary).
4. Add an optional output mode to save extracted tables as CSV/JSON for reproducibility.

## 3) Accuracy Improvements (high impact)

1. Tune thresholds independently on a validation set:
	- `table_threshold` grid (e.g., 0.4 to 0.8).
	- `cell_threshold` grid (e.g., 0.4 to 0.8).
2. Improve OCR preprocessing per cell:
	- Grayscale conversion.
	- Contrast/threshold normalization.
	- Optional deskew for rotated scans.
3. Improve row/column assignment:
	- Infer column anchors from x-clusters.
	- Snap cells to nearest anchor.
	- Handle merged/missing cells more explicitly.
4. Strengthen header detection heuristic and keep a fallback to default numeric columns.

## 4) Performance and Scale

1. Add optional device selection (`cpu`/`cuda`) and explicit model/tensor placement.
2. Add optional per-page parallelism for large PDFs.
3. Profile OCR bottlenecks and consider batching or alternative OCR settings.

## 5) Testing and Quality Gates

1. Build a small regression corpus of PDFs (clean scans, noisy scans, merged cells, multi-image pages).
2. Add smoke tests for:
	- No-image pages.
	- Corrupt embedded images.
	- Empty OCR cells.
3. Track metrics per run:
	- Tables detected/page.
	- Cells detected/table.
	- Non-empty OCR ratio.
	- Runtime/page.

## 6) Suggested Execution Order

1. Environment + baseline run.
2. Logging + CLI validation + metadata output.
3. Threshold tuning on validation PDFs.
4. OCR preprocessing and row/column anchoring improvements.
5. Performance optimization and regression test automation.
