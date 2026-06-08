import argparse
from pathlib import Path
from typing import Any

from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel


DEFAULT_MODEL_BASE_DIR = Path(__file__).resolve().parent / "local_models"


REQUIRED_MODELS = {
	"detection": "microsoft/table-transformer-detection",
	"structure": "microsoft/table-transformer-structure-recognition-v1.1-all",
	"ocr": "microsoft/trocr-base-printed",
}


def _is_model_present(model_dir: Path) -> bool:
	"""Treat a model as present when core config files already exist locally."""
	required_files = ["config.json", "preprocessor_config.json"]
	return model_dir.exists() and all((model_dir / file_name).exists() for file_name in required_files)


def _ensure_model_local(model_id: str, model_dir: Path, processor_cls: Any, model_cls: Any) -> None:
	if _is_model_present(model_dir):
		print(f"[skip] {model_id} already present at {model_dir}")
		return

	print(f"[download] Pulling {model_id} into {model_dir} ...")
	model_dir.mkdir(parents=True, exist_ok=True)

	processor = processor_cls.from_pretrained(model_id)
	model = model_cls.from_pretrained(model_id)

	processor.save_pretrained(model_dir)
	model.save_pretrained(model_dir)
	print(f"[ok] Saved {model_id} to {model_dir}")


def ensure_required_models(base_dir: Path) -> tuple[Path, Path, Path]:
	"""Ensure detection, structure, and OCR models exist locally under base_dir."""
	detect_dir = base_dir / "table_transformer_detection_local"
	struct_dir = base_dir / "table_transformer_structure_local"
	ocr_dir = base_dir / "trocr_base_printed_local"

	_ensure_model_local(
		REQUIRED_MODELS["detection"],
		detect_dir,
		AutoImageProcessor,
		AutoModelForObjectDetection,
	)
	_ensure_model_local(
		REQUIRED_MODELS["structure"],
		struct_dir,
		AutoImageProcessor,
		AutoModelForObjectDetection,
	)
	_ensure_model_local(
		REQUIRED_MODELS["ocr"],
		ocr_dir,
		TrOCRProcessor,
		VisionEncoderDecoderModel,
	)

	return detect_dir, struct_dir, ocr_dir


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Download and cache local table-transformer models required by example_1.py"
	)
	parser.add_argument(
		"--base-dir",
		default=str(DEFAULT_MODEL_BASE_DIR),
		help="Directory where local model folders are stored.",
	)
	args = parser.parse_args()

	base_dir = Path(args.base_dir).resolve()
	detect_dir, struct_dir, ocr_dir = ensure_required_models(base_dir)

	print("\nReady to use these paths with example_1.py:")
	print(f"--detect-model-dir \"{detect_dir}\"")
	print(f"--struct-model-dir \"{struct_dir}\"")
	print(f"--ocr-model-dir \"{ocr_dir}\"")


if __name__ == "__main__":
	main()