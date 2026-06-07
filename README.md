# local-ocr

Extracts tables from image-based PDFs into one Excel workbook with one table per sheet.

## What this project includes

- `model_bootstrap.py`: downloads and stores required models locally.
- `example_1.py`: runs table detection, structure detection, OCR, and writes Excel output.

## Usage

### One-time model bootstrap (online)

```powershell
python model_bootstrap.py
```

This populates `local_models/` with:

- Table Transformer detection model
- Table Transformer structure model
- EasyOCR model files

### Run extraction (offline)

```powershell
python example_1.py --pdf-path "C:\path\to\input.pdf" --output-xlsx "C:\path\to\tables.xlsx"
```

Output behavior:

- One Excel file is produced.
- One extracted table per sheet.
- Sheet names are `page_<N>_table_<M>`.

### Optional flags

- `--table-threshold` (default: `0.6`)
- `--cell-threshold` (default: `0.6`)
- `--no-promote-header`
- `--detect-model-dir`, `--struct-model-dir`, `--ocr-model-dir`

## Typical workflow

1. Run model bootstrap once while connected to the internet.
2. Move project (including `local_models/`) to offline environment if needed.
3. Run `example_1.py` on PDFs and collect `.xlsx` outputs.
