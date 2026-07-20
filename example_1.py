import io
import os
import argparse
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

import torch
import pandas as pd
from pypdf import PdfReader
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


DEFAULT_MODEL_BASE_DIR = Path(__file__).resolve().parent / "local_models"
DEFAULT_DETECT_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "table_transformer_detection_local"
DEFAULT_STRUCT_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "table_transformer_structure_local"
DEFAULT_TROCR_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "trocr_base_printed_local"


ModelPath = str | os.PathLike[str]


class ExtractedTable(TypedDict):
    page: int
    table_on_page: int
    dataframe: pd.DataFrame


def _load_image_processor(model_path: ModelPath) -> Any:
    return AutoImageProcessor.from_pretrained(model_path, local_files_only=True)  # type: ignore[reportUnknownMemberType]


def _processor_call(processor: Any, image: Image.Image) -> Any:
    """Call an image processor, working around transformers < 4.37 not accepting
    a size dict with only 'longest_edge'.  In newer transformers that format means
    "scale so the longest edge equals N" (upscale or downscale).  We replicate
    that behaviour explicitly and then tell the processor to skip its own resize."""
    size = getattr(processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size and "shortest_edge" not in size:
        longest_edge = size["longest_edge"]
        img_w, img_h = image.size  # PIL: (width, height)
        scale = longest_edge / max(img_w, img_h)
        if scale != 1.0:
            new_w = max(1, round(img_w * scale))
            new_h = max(1, round(img_h * scale))
            image = image.resize((new_w, new_h), Image.BILINEAR)
        return processor(images=image, return_tensors="pt", do_resize=False)
    return processor(images=image, return_tensors="pt")


def _load_object_detection_model(model_path: ModelPath) -> Any:
    return AutoModelForObjectDetection.from_pretrained(  # type: ignore[reportUnknownMemberType]
        model_path,
        local_files_only=True,
        use_pretrained_backbone=False,
    )


def _load_trocr_processor(model_path: ModelPath) -> TrOCRProcessor:
    return TrOCRProcessor.from_pretrained(model_path, local_files_only=True)  # type: ignore[reportUnknownMemberType]


def _load_trocr_model(model_path: ModelPath) -> VisionEncoderDecoderModel:
    return cast(
        VisionEncoderDecoderModel,
        VisionEncoderDecoderModel.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            local_files_only=True,
        ),
    )


def _open_rgb_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")  # type: ignore[reportUnknownMemberType]


def _crop_image(image: Image.Image, box: Sequence[int]) -> Image.Image:
    return image.crop(tuple(box))  # type: ignore[reportUnknownMemberType]


def _image_size(image: Image.Image) -> tuple[int, int]:
    image_obj = cast(Any, image)
    return int(image_obj.width), int(image_obj.height)


def _processor_pixel_values(processor: TrOCRProcessor, image: Image.Image) -> torch.Tensor:
    processor_obj = cast(Any, processor)
    processor_output: Any = processor_obj(images=image, return_tensors="pt")
    return cast(torch.Tensor, processor_output.pixel_values)


def _generate_ids(model: VisionEncoderDecoderModel, pixel_values: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, model.generate(pixel_values))  # type: ignore[reportUnknownMemberType]


def _decode_text(processor: TrOCRProcessor, generated_ids: torch.Tensor) -> list[str]:
    processor_obj = cast(Any, processor)
    return cast(
        list[str],
        processor_obj.batch_decode(generated_ids, skip_special_tokens=True),
    )

class OfflineTableExtractorPipeline:
    def __init__(
        self,
        detect_model_path: ModelPath,
        structure_model_path: ModelPath,
        ocr_model_path: ModelPath,
    ) -> None:
        print("[1/4] Initializing local TrOCR framework...")
        self.ocr_processor: TrOCRProcessor = _load_trocr_processor(ocr_model_path)
        self.ocr_model: VisionEncoderDecoderModel = _load_trocr_model(ocr_model_path)
        self.ocr_model.eval()

        print("[2/4] Loading Stage 1: Table Detection Model...")
        self.detect_processor: Any = _load_image_processor(detect_model_path)
        self.detect_model: Any = _load_object_detection_model(detect_model_path)

        print("[3/4] Loading Stage 2: Structure Recognition Model...")
        self.struct_processor: Any = _load_image_processor(structure_model_path)
        self.struct_model: Any = _load_object_detection_model(structure_model_path)
        print("--- All local models loaded successfully and locked offline ---")

    def _ocr_cell_text(self, cell_image: Image.Image) -> str:
        """Run OCR on a PIL cell image using a local TrOCR model."""
        pixel_values = _processor_pixel_values(self.ocr_processor, cell_image)
        with torch.no_grad():
            generated_ids = _generate_ids(self.ocr_model, pixel_values)
        decoded = _decode_text(self.ocr_processor, generated_ids)
        text = decoded[0] if decoded else ""
        return text.strip()

    @staticmethod
    def _vertical_overlap_ratio(box_a: list[int], box_b: list[int]) -> float:
        """Return the fraction of the shorter box's height that overlaps vertically."""
        overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        shorter = min(box_a[3] - box_a[1], box_b[3] - box_b[1])
        return overlap / shorter if shorter > 0 else 0.0

    @staticmethod
    def _clip_box(box: Sequence[float], width: int, height: int) -> list[int] | None:
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
    def _promote_header(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
        """Optionally promote first row to header only when it looks like a header."""
        if not enabled or df.empty or len(df) <= 1:
            return df

        header_values = [str(value).strip() for value in df.iloc[0].tolist()]
        non_empty = [value for value in header_values if value]
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

        new_cols: list[str] = []
        seen: dict[str, int] = {}
        for idx, raw in enumerate(header_values, start=1):
            name = raw or f"Column_{idx}"
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
        pdf_path: ModelPath,
        table_threshold: float = 0.6,
        cell_threshold: float = 0.6,
        promote_header: bool = True,
    ) -> list[ExtractedTable]:
        pdf_source = Path(pdf_path)
        if not pdf_source.exists():
            print(f"Error: Target file not found at {pdf_source}")
            return []

        reader = PdfReader(pdf_source)
        all_extracted_tables: list[ExtractedTable] = []

        for page_num, page in enumerate(reader.pages):
            if not page.images:
                print(f"Page {page_num + 1}: Skipping (No embedded scanned images).")
                continue

            print(f"\n--- Processing Page {page_num + 1} ---")
            page_table_counter = 0

            for img_idx, scanned_image_obj in enumerate(page.images):
                try:
                    # 1. Grab raw image bytes out of PDF envelope
                    full_page_image = _open_rgb_image(scanned_image_obj.data)
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
                    detect_inputs = _processor_call(self.detect_processor, full_page_image)
                    with torch.no_grad():
                        detect_outputs = self.detect_model(**detect_inputs)
                except Exception as ex:
                    print(
                        f"  Image #{img_idx + 1}: Detection stage failed on "
                        f"Page {page_num + 1} ({ex})."
                    )
                    continue

                width, height = _image_size(full_page_image)
                detect_sizes = torch.tensor([(height, width)])
                detect_results = self.detect_processor.post_process_object_detection(
                    detect_outputs, threshold=table_threshold, target_sizes=detect_sizes
                )[0]

                # Filter detections to grab macro table elements (Label 0)
                table_boxes: list[list[int]] = []
                for label, box in zip(detect_results["labels"], detect_results["boxes"]):
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
                    cropped_table_img = _crop_image(full_page_image, t_box)

                    # ==========================================
                    # STAGE 2: PARSE THE INTERNAL TABLE CELLS
                    # ==========================================
                    try:
                        struct_inputs = _processor_call(self.struct_processor, cropped_table_img)
                        with torch.no_grad():
                            struct_outputs = self.struct_model(**struct_inputs)
                    except Exception as ex:
                        print(
                            f"    Table Block #{t_idx + 1}: Structure stage failed "
                            f"({ex})."
                        )
                        continue

                    c_width, c_height = _image_size(cropped_table_img)
                    struct_sizes = torch.tensor([(c_height, c_width)])
                    struct_results = self.struct_processor.post_process_object_detection(
                        struct_outputs, threshold=cell_threshold, target_sizes=struct_sizes
                    )[0]

                    # 3. Collect row boxes from the structure model.
                    #    Label 3 = table column header (always a row).
                    #    Label 2 = table row (skip any that duplicate a label-3 region).
                    #    The model often tags the header area with both labels, so we
                    #    give label-3 precedence and drop overlapping label-2 boxes to
                    #    avoid producing a duplicate header row.
                    #    Label 1 = table column.
                    header_boxes: list[list[int]] = []
                    data_row_boxes: list[list[int]] = []
                    col_boxes: list[list[int]] = []
                    for label, box in zip(struct_results["labels"], struct_results["boxes"]):
                        lv = label.item()
                        if lv == 3:  # table column header
                            clipped = self._clip_box(box.tolist(), c_width, c_height)
                            if clipped is not None:
                                header_boxes.append(clipped)
                        elif lv == 2:  # table row
                            clipped = self._clip_box(box.tolist(), c_width, c_height)
                            if clipped is not None:
                                data_row_boxes.append(clipped)
                        elif lv == 1:  # table column
                            clipped = self._clip_box(box.tolist(), c_width, c_height)
                            if clipped is not None:
                                col_boxes.append(clipped)

                    # Drop label-2 rows that substantially overlap a label-3 header row
                    # to prevent the header from appearing twice in the output grid.
                    filtered_data_rows = [
                        rb for rb in data_row_boxes
                        if not any(
                            self._vertical_overlap_ratio(rb, hb) > 0.5
                            for hb in header_boxes
                        )
                    ]
                    row_boxes = header_boxes + filtered_data_rows

                    if not row_boxes or not col_boxes:
                        continue

                    # Sort rows top-to-bottom, columns left-to-right
                    row_boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
                    col_boxes.sort(key=lambda b: (b[0] + b[2]) / 2)

                    # Build the grid by OCR-ing each row × column intersection
                    matrix_grid: list[list[str]] = []
                    for rbox in row_boxes:
                        row_strings: list[str] = []
                        for cbox in col_boxes:
                            x0 = max(rbox[0], cbox[0])
                            y0 = max(rbox[1], cbox[1])
                            x1 = min(rbox[2], cbox[2])
                            y1 = min(rbox[3], cbox[3])
                            if x1 <= x0 or y1 <= y0:
                                row_strings.append("")
                                continue
                            cell_crop = _crop_image(cropped_table_img, [x0, y0, x1, y1])
                            try:
                                cell_string = self._ocr_cell_text(cell_crop)
                            except Exception as ex:
                                print(
                                    f"      OCR failed for a cell in Table Block #{t_idx + 1} "
                                    f"({ex})."
                                )
                                cell_string = ""
                            row_strings.append(cell_string)
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
        default=str(DEFAULT_TROCR_MODEL_DIR),
        help="Local directory containing TrOCR model files.",
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
        with cast(Any, pd.ExcelWriter(args.output_xlsx)) as writer:
            for item in results:
                sheet_name = f"page_{item['page']}_table_{item['table_on_page']}"
                # Excel sheet names are limited to 31 characters.
                sheet_name = sheet_name[:31]
                cast(Any, item["dataframe"]).to_excel(writer, sheet_name=sheet_name, index=False)
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
