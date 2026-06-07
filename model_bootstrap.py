import argparse
from pathlib import Path

import easyocr
from transformers import AutoImageProcessor, AutoModelForObjectDetection


DEFAULT_MODEL_BASE_DIR = Path(__file__).resolve().parent / "local_models"
DEFAULT_EASYOCR_MODEL_DIR = DEFAULT_MODEL_BASE_DIR / "easyocr"


REQUIRED_MODELS = {
	"detection": "microsoft/table-transformer-detection",
	"structure": "microsoft/table-transformer-structure-recognition-v1.1-all",
}


def _is_model_present(model_dir: Path) -> bool:
	"""Treat a model as present when core config files already exist locally."""
	required_files = ["config.json", "preprocessor_config.json"]
	return model_dir.exists() and all((model_dir / file_name).exists() for file_name in required_files)


def _ensure_model_local(model_id: str, model_dir: Path) -> None:
	if _is_model_present(model_dir):
		print(f"[skip] {model_id} already present at {model_dir}")
		return

	print(f"[download] Pulling {model_id} into {model_dir} ...")
	model_dir.mkdir(parents=True, exist_ok=True)

	processor = AutoImageProcessor.from_pretrained(model_id)
	model = AutoModelForObjectDetection.from_pretrained(model_id)

	processor.save_pretrained(model_dir)
	model.save_pretrained(model_dir)
	print(f"[ok] Saved {model_id} to {model_dir}")


def ensure_required_models(base_dir: Path) -> tuple[Path, Path]:
	"""Ensure both models used by example_1.py exist locally under base_dir."""
	detect_dir = base_dir / "table_transformer_detection_local"
	struct_dir = base_dir / "table_transformer_structure_local"

	_ensure_model_local(REQUIRED_MODELS["detection"], detect_dir)
	_ensure_model_local(REQUIRED_MODELS["structure"], struct_dir)

	return detect_dir, struct_dir


def ensure_easyocr_models(model_dir: Path) -> Path:
	"""Pre-download EasyOCR weights into a project-local directory for offline runtime."""
	model_dir.mkdir(parents=True, exist_ok=True)
	print(f"[bootstrap] Ensuring EasyOCR models are present at {model_dir} ...")
	easyocr.Reader(
		["en"],
		gpu=False,
		model_storage_directory=str(model_dir),
		download_enabled=True,
	)
	print(f"[ok] EasyOCR models ready at {model_dir}")
	return model_dir


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
	detect_dir, struct_dir = ensure_required_models(base_dir)
	ocr_dir = ensure_easyocr_models(base_dir / DEFAULT_EASYOCR_MODEL_DIR.name)

	print("\nReady to use these paths with example_1.py:")
	print(f"--detect-model-dir \"{detect_dir}\"")
	print(f"--struct-model-dir \"{struct_dir}\"")
	print(f"--ocr-model-dir \"{ocr_dir}\"")


if __name__ == "__main__":
	main()