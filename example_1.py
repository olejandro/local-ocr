import io
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from pypdf import PdfReader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import easyocr


DEFAULT_MODEL_BASE_DIR = Path(__file__).resolve().parent / "local_models"
DEFAULT_DETECT_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "table_transformer_detection_local"
DEFAULT_STRUCT_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "table_transformer_structure_local"
DEFAULT_EASYOCR_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "easyocr"

class OfflineTableExtractorPipeline:
    def __init__(self, detect_model_path, structure_model_path, ocr_model_path):
        print("[1/4] Initializing local EasyOCR framework...")
        self.ocr_reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=str(ocr_model_path),
            download_enabled=False,
        )
        
        print("[2/4] Loading Stage 1: Table Detection Model...")
        self.detect_processor = AutoImageProcessor.from_pretrained(detect_model_path, local_files_only=True)
        self.detect_model = AutoModelForObjectDetection.from_pretrained(
            detect_model_path, local_files_only=True, use_pretrained_backbone=False
        )
        
        print("[3/4] Loading Stage 2: Structure Recognition Model...")
        self.struct_processor = AutoImageProcessor.from_pretrained(structure_model_path, local_files_only=True)
        self.struct_model = AutoModelForObjectDetection.from_pretrained(
            structure_model_path, local_files_only=True, use_pretrained_backbone=False
        )
        print("--- All local models loaded successfully and locked offline ---")

    @staticmethod
    def _clip_box(box, width, height):
        """Clamp a predicted box into image bounds and reject zero-area boxes."""
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    @staticmethod
    def _promote_header(df, enabled=True):
        """Optionally promote first row to header only when it looks like a header."""
        if not enabled or df.empty or len(df) <= 1:
            return df

        header_candidate = df.iloc[0].fillna("").astype(str)
        non_empty = [v.strip() for v in header_candidate.tolist() if v.strip()]
        if not non_empty:
            return df

        # Heuristic: header-like rows usually contain mostly non-numeric strings.
        numeric_like = 0
        for value in non_empty:
            try:
                float(value.replace(",", ""))
                numeric_like += 1
            except ValueError:
                pass

        is_header_like = (numeric_like / len(non_empty)) < 0.5
        if not is_header_like:
            return df

        new_cols = []
        seen = {}
        for idx, raw in enumerate(header_candidate.tolist(), start=1):
            name = str(raw).strip() or f"Column_{idx}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            new_cols.append(name)

        out = df[1:].reset_index(drop=True)
        out.columns = new_cols
        return out

    def extract_tables_from_pdf(
        self,
        pdf_path,
        table_threshold=0.6,
        cell_threshold=0.6,
        promote_header=True,
    ):
        if not os.path.exists(pdf_path):
            print(f"Error: Target file not found at {pdf_path}")
            return []

        reader = PdfReader(pdf_path)
        all_extracted_tables = []

        for page_num, page in enumerate(reader.pages):
            if not page.images:
                print(f"Page {page_num + 1}: Skipping (No embedded scanned images).")
                continue

            print(f"\n--- Processing Page {page_num + 1} ---")
            page_table_counter = 0

            for img_idx, scanned_image_obj in enumerate(page.images):
                try:
                    # 1. Grab raw image bytes out of PDF envelope
                    full_page_image = Image.open(io.BytesIO(scanned_image_obj.data)).convert("RGB")
                except Exception as ex:
                    print(
                        f"  Image #{img_idx + 1}: Failed to decode embedded image on "
                        f"Page {page_num + 1} ({ex})."
                    )
                    continue

                # ==========================================
                # STAGE 1: LOCATE MACRO TABLES ON THE PAGE IMAGE
                # ==========================================
                try:
                    detect_inputs = self.detect_processor(images=full_page_image, return_tensors="pt")
                    with torch.no_grad():
                        detect_outputs = self.detect_model(**detect_inputs)
                except Exception as ex:
                    print(
                        f"  Image #{img_idx + 1}: Detection stage failed on "
                        f"Page {page_num + 1} ({ex})."
                    )
                    continue

                detect_sizes = torch.tensor([full_page_image.size[::-1]])
                detect_results = self.detect_processor.post_process_object_detection(
                    detect_outputs, threshold=table_threshold, target_sizes=detect_sizes
                )[0]

                # Filter detections to grab macro table elements (Label 0)
                table_boxes = []
                width, height = full_page_image.size
                for score, label, box in zip(
                    detect_results["scores"], detect_results["labels"], detect_results["boxes"]
                ):
                    if label.item() == 0:  # 0 is the label for a complete macro table block
                        clipped = self._clip_box(box.tolist(), width, height)
                        if clipped is not None:
                            table_boxes.append(clipped)

                print(
                    f"  Image #{img_idx + 1}: Found {len(table_boxes)} macro table(s) "
                    f"on Page {page_num + 1}."
                )

                # Process each individual macro table discovered on the image
                for t_idx, t_box in enumerate(table_boxes):
                    print(f"    Processing Table Block #{t_idx + 1}/{len(table_boxes)}...")

                    # 2. Crop full page directly down to just the isolated table bounds
                    cropped_table_img = full_page_image.crop(t_box)

                    # ==========================================
                    # STAGE 2: PARSE THE INTERNAL TABLE CELLS
                    # ==========================================
                    try:
                        struct_inputs = self.struct_processor(images=cropped_table_img, return_tensors="pt")
                        with torch.no_grad():
                            struct_outputs = self.struct_model(**struct_inputs)
                    except Exception as ex:
                        print(
                            f"    Table Block #{t_idx + 1}: Structure stage failed "
                            f"({ex})."
                        )
                        continue

                    struct_sizes = torch.tensor([cropped_table_img.size[::-1]])
                    struct_results = self.struct_processor.post_process_object_detection(
                        struct_outputs, threshold=cell_threshold, target_sizes=struct_sizes
                    )[0]

                    detected_cells = []
                    c_width, c_height = cropped_table_img.size
                    for score, label, box in zip(
                        struct_results["scores"], struct_results["labels"], struct_results["boxes"]
                    ):
                        if label.item() != 4:  # Label 4 = Individual Table Cells
                            continue

                        box_coords = self._clip_box(box.tolist(), c_width, c_height)
                        if box_coords is None:
                            continue

                        # Compute geometric properties relative to cropped table window
                        x_center = (box_coords[0] + box_coords[2]) / 2
                        y_center = (box_coords[1] + box_coords[3]) / 2
                        cell_height = box_coords[3] - box_coords[1]

                        # Isolate text elements inside cell via native local OCR execution
                        cell_crop = cropped_table_img.crop(box_coords)
                        cell_np = np.array(cell_crop)
                        try:
                            cell_text_list = self.ocr_reader.readtext(cell_np, detail=0)
                        except Exception as ex:
                            print(
                                f"      OCR failed for a cell in Table Block #{t_idx + 1} "
                                f"({ex})."
                            )
                            cell_text_list = []
                        cell_string = " ".join(cell_text_list).strip()

                        detected_cells.append(
                            {
                                "x": x_center,
                                "y": y_center,
                                "text": cell_string,
                                "height": cell_height,
                            }
                        )

                    if not detected_cells:
                        continue

                    # 3. Reconstruct extracted array into spatial table row assignments
                    detected_cells.sort(key=lambda c: c["y"])
                    median_height = np.median([c["height"] for c in detected_cells])
                    y_tolerance = max(4.0, median_height * 0.45)

                    rows = []
                    current_row = [detected_cells[0]]
                    current_row_y = detected_cells[0]["y"]

                    for cell in detected_cells[1:]:
                        if abs(cell["y"] - current_row_y) <= y_tolerance:
                            current_row.append(cell)
                            current_row_y = float(np.mean([c["y"] for c in current_row]))
                        else:
                            current_row.sort(key=lambda c: c["x"])
                            rows.append(current_row)
                            current_row = [cell]
                            current_row_y = cell["y"]
                    current_row.sort(key=lambda c: c["x"])
                    rows.append(current_row)

                    # Normalize alignment array constraints into standard grid shapes
                    max_cols = max(len(r) for r in rows)
                    matrix_grid = []
                    for r in rows:
                        row_strings = [cell["text"] for cell in r]
                        while len(row_strings) < max_cols:
                            row_strings.append("")
                        matrix_grid.append(row_strings)

                    # Create final presentation Pandas Dataframe container
                    df = pd.DataFrame(matrix_grid)
                    df = self._promote_header(df, enabled=promote_header)
                    page_table_counter += 1
                    all_extracted_tables.append(
                        {
                            "page": page_num + 1,
                            "table_on_page": page_table_counter,
                            "dataframe": df,
                        }
                    )

        return all_extracted_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline PDF table extraction using local detection, structure, and OCR models."
    )
    parser.add_argument(
        "--detect-model-dir",
        default=str(DEFAULT_DETECT_MODEL_DIR),
        help="Local directory for the table detection model.",
    )
    parser.add_argument(
        "--struct-model-dir",
        default=str(DEFAULT_STRUCT_MODEL_DIR),
        help="Local directory for the table structure model.",
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to the target input PDF.",
    )
    parser.add_argument(
        "--ocr-model-dir",
        default=str(DEFAULT_EASYOCR_MODEL_DIR),
        help="Local directory containing EasyOCR model weights.",
    )
    parser.add_argument(
        "--table-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for stage-1 macro table detection.",
    )
    parser.add_argument(
        "--cell-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for stage-2 cell detection.",
    )
    parser.add_argument(
        "--no-promote-header",
        action="store_true",
        help="Disable first-row-to-header promotion heuristic.",
    )
    parser.add_argument(
        "--output-xlsx",
        default="extracted_tables.xlsx",
        help="Output Excel workbook path (one table per sheet).",
    )
    args = parser.parse_args()

    # Spin up execution instance
    pipeline = OfflineTableExtractorPipeline(
        args.detect_model_dir,
        args.struct_model_dir,
        args.ocr_model_dir,
    )
    results = pipeline.extract_tables_from_pdf(
        args.pdf_path,
        table_threshold=args.table_threshold,
        cell_threshold=args.cell_threshold,
        promote_header=not args.no_promote_header,
    )

    if results:
        with pd.ExcelWriter(args.output_xlsx) as writer:
            for item in results:
                sheet_name = f"page_{item['page']}_table_{item['table_on_page']}"
                # Excel sheet names are limited to 31 characters.
                sheet_name = sheet_name[:31]
                item["dataframe"].to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved {len(results)} table(s) to Excel file: {args.output_xlsx}")
    else:
        print("No tables were extracted; no Excel file was created.")

    # Render final console spreadsheet printouts
    for index, item in enumerate(results):
        df = item["dataframe"]
        print("\n==============================================")
        print(
            f"FINAL PASSED DATAFRAME BLOCK #{index + 1} "
            f"(page={item['page']}, table={item['table_on_page']})"
        )
        print("==============================================")
        print(df.to_string())
