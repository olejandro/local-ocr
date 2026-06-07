import io
import os
import torch
import numpy as np
import pandas as pd
from pypdf import PdfReader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import easyocr

class OfflineTableExtractorPipeline:
    def __init__(self, detect_model_path, structure_model_path):
        print("[1/4] Initializing local EasyOCR framework...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        
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

    def extract_tables_from_pdf(self, pdf_path, threshold=0.6):
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
            
            # 1. Grab raw image bytes out of PDF envelope
            scanned_image_obj = page.images[0]
            full_page_image = Image.open(io.BytesIO(scanned_image_obj.data)).convert("RGB")
            
            # ==========================================
            # STAGE 1: LOCATE MACRO TABLES ON THE PAGE
            # ==========================================
            detect_inputs = self.detect_processor(images=full_page_image, return_tensors="pt")
            with torch.no_grad():
                detect_outputs = self.detect_model(**detect_inputs)
                
            detect_sizes = torch.tensor([full_page_image.size[::-1]])
            detect_results = self.detect_processor.post_process_object_detection(
                detect_outputs, threshold=threshold, target_sizes=detect_sizes
            )[0]
            
            # Filter detections to grab macro table elements (Label 0)
            table_boxes = []
            for score, label, box in zip(detect_results["scores"], detect_results["labels"], detect_results["boxes"]):
                if label.item() == 0:  # 0 is the label for a complete macro table block
                    table_boxes.append([round(c) for c in box.tolist()])

            print(f"Found {len(table_boxes)} macro table(s) on Page {page_num + 1}.")

            # Process each individual macro table discovered on the page
            for t_idx, t_box in enumerate(table_boxes):
                print(f"  Processing Table Block #{t_idx + 1}/{len(table_boxes)}...")
                
                # 2. Crop full page directly down to just the isolated table bounds
                cropped_table_img = full_page_image.crop(t_box)
                
                # ==========================================
                # STAGE 2: PARSE THE INTERNAL TABLE CELLS
                # ==========================================
                struct_inputs = self.struct_processor(images=cropped_table_img, return_tensors="pt")
                with torch.no_grad():
                    struct_outputs = self.struct_model(**struct_inputs)
                    
                struct_sizes = torch.tensor([cropped_table_img.size[::-1]])
                struct_results = self.struct_processor.post_process_object_detection(
                    struct_outputs, threshold=threshold, target_sizes=struct_sizes
                )[0]
                
                detected_cells = []
                for score, label, box in zip(struct_results["scores"], struct_results["labels"], struct_results["boxes"]):
                    box_coords = [round(c) for c in box.tolist()]
                    
                    if label.item() == 4:  # Label 4 = Individual Table Cells
                        # Compute geometric properties relative to cropped table window
                        x_center = (box_coords[0] + box_coords[2]) / 2
                        y_center = (box_coords[1] + box_coords[3]) / 2
                        cell_height = box_coords[3] - box_coords[1]
                        
                        # Isolate text elements inside cell via native local OCR execution
                        cell_crop = cropped_table_img.crop(box_coords)
                        cell_np = np.array(cell_crop)
                        cell_text_list = self.ocr_reader.readtext(cell_np, detail=0)
                        cell_string = " ".join(cell_text_list).strip()
                        
                        detected_cells.append({
                            "x": x_center,
                            "y": y_center,
                            "text": cell_string,
                            "height": cell_height
                        })
                
                if not detected_cells:
                    continue

                # 3. Reconstruct extracted array into spatial table row assignments
                detected_cells.sort(key=lambda c: c["y"])
                avg_height = np.mean([c["height"] for c in detected_cells])
                y_tolerance = avg_height * 0.4  # Group values dynamically into row groups
                
                rows = []
                current_row = [detected_cells[0]]
                
                for cell in detected_cells[1:]:
                    if abs(cell["y"] - current_row[-1]["y"]) <= y_tolerance:
                        current_row.append(cell)
                    else:
                        current_row.sort(key=lambda c: c["x"])
                        rows.append(current_row)
                        current_row = [cell]
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
                if not df.empty and len(df) > 1:
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
                    
                all_extracted_tables.append(df)

        return all_extracted_tables


if __name__ == "__main__":
    # --- Configuration Window Mapping Paths ---
    DETECT_MODEL_DIR = "C:\\Models\\table_transformer_detection_local"
    STRUCT_MODEL_DIR = "C:\\Models\\table_transformer_structure_local"
    TARGET_PDF_FILE = "C:\\Users\\YourUsername\\Documents\\scanned_invoice.pdf"
    
    # Spin up execution instance
    pipeline = OfflineTableExtractorPipeline(DETECT_MODEL_DIR, STRUCT_MODEL_DIR)
    results = pipeline.extract_tables_from_pdf(TARGET_PDF_FILE)
    
    # Render final console spreadsheet printouts
    for index, df in enumerate(results):
        print(f"\n==============================================")
        print(f"FINAL PASSED DATAFRAME BLOCK #{index + 1}")
        print(f"==============================================")
        print(df.to_string())
