import importlib
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


class _FakeDataFrame:
    def __init__(self, data):
        self.data = data


class _FakeTorch(types.ModuleType):
    class Tensor:
        pass

    @staticmethod
    def tensor(values):
        return values

    @staticmethod
    @contextmanager
    def no_grad():
        yield


class _FakeImageModule(types.ModuleType):
    class Image:
        pass


class _FakePdfReader:
    pages = []

    def __init__(self, _):
        self.pages = list(type(self).pages)


class _FakeModelFactory:
    @staticmethod
    def from_pretrained(*_, **__):
        return object()


class _FakeVisionEncoderDecoderModel(_FakeModelFactory):
    def eval(self):
        return None

    def generate(self, _):
        return []


class _TensorScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _TensorBox:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeImage:
    def __init__(self, width=200, height=100):
        self.width = width
        self.height = height

    def crop(self, _):
        return _FakeImage(width=100, height=50)


class _FakePage:
    def __init__(self, images):
        self.images = images


class _FakeEmbeddedImage:
    def __init__(self, data):
        self.data = data


class _FakeProcessor:
    def __init__(self, results):
        self._results = results

    def __call__(self, **_):
        return {}

    def post_process_object_detection(self, *_args, **_kwargs):
        return [self._results]


class OfflineTableExtractorPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_numpy = types.ModuleType("numpy")

        def _median(values):
            sorted_values = sorted(values)
            middle = len(sorted_values) // 2
            if len(sorted_values) % 2 == 1:
                return sorted_values[middle]
            return (sorted_values[middle - 1] + sorted_values[middle]) / 2

        fake_numpy.median = _median
        fake_numpy.mean = lambda values: sum(values) / len(values)

        fake_pandas = types.ModuleType("pandas")
        fake_pandas.DataFrame = _FakeDataFrame

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = _FakePdfReader

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoImageProcessor = _FakeModelFactory
        fake_transformers.AutoModelForObjectDetection = _FakeModelFactory
        fake_transformers.TrOCRProcessor = _FakeModelFactory
        fake_transformers.VisionEncoderDecoderModel = _FakeVisionEncoderDecoderModel

        fake_pil = types.ModuleType("PIL")
        fake_pil_image = _FakeImageModule("PIL.Image")
        fake_pil.Image = fake_pil_image

        cls._module_patcher = mock.patch.dict(
            sys.modules,
            {
                "torch": _FakeTorch("torch"),
                "numpy": fake_numpy,
                "pandas": fake_pandas,
                "pypdf": fake_pypdf,
                "transformers": fake_transformers,
                "PIL": fake_pil,
                "PIL.Image": fake_pil_image,
            },
        )
        cls._module_patcher.start()
        cls.example_1 = importlib.import_module("example_1")

    @classmethod
    def tearDownClass(cls):
        cls._module_patcher.stop()

    def _create_test_pipeline(self):
        pipeline = object.__new__(self.example_1.OfflineTableExtractorPipeline)
        pipeline.detect_processor = _FakeProcessor({"labels": [], "boxes": []})
        pipeline.detect_model = lambda **_: {}
        pipeline.struct_processor = _FakeProcessor({"labels": [], "boxes": []})
        pipeline.struct_model = lambda **_: {}
        pipeline._ocr_cell_text = lambda _img: ""
        return pipeline

    def test_extract_tables_returns_empty_when_pages_have_no_images(self):
        _FakePdfReader.pages = [_FakePage(images=[])]
        pipeline = self._create_test_pipeline()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            results = pipeline.extract_tables_from_pdf(f.name)

        self.assertEqual(results, [])

    def test_extract_tables_skips_corrupt_page_images(self):
        _FakePdfReader.pages = [_FakePage(images=[_FakeEmbeddedImage(data=b"broken")])]
        pipeline = self._create_test_pipeline()

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as f,
            mock.patch.object(
                self.example_1,
                "_open_rgb_image",
                side_effect=ValueError("cannot decode image"),
            ),
        ):
            results = pipeline.extract_tables_from_pdf(f.name)

        self.assertEqual(results, [])

    def test_extract_tables_builds_matrix_from_detected_cells(self):
        _FakePdfReader.pages = [_FakePage(images=[_FakeEmbeddedImage(data=b"ok")])]
        pipeline = self._create_test_pipeline()
        pipeline.detect_processor = _FakeProcessor(
            {
                "labels": [_TensorScalar(0)],
                "boxes": [_TensorBox([0, 0, 100, 50])],
            }
        )
        pipeline.struct_processor = _FakeProcessor(
            {
                "labels": [
                    _TensorScalar(4),
                    _TensorScalar(4),
                    _TensorScalar(4),
                    _TensorScalar(4),
                ],
                "boxes": [
                    _TensorBox([0, 0, 40, 10]),
                    _TensorBox([50, 0, 90, 10]),
                    _TensorBox([0, 20, 40, 30]),
                    _TensorBox([50, 20, 90, 30]),
                ],
            }
        )

        pipeline._ocr_cell_text = mock.Mock(side_effect=["A1", "B1", "A2", "B2"])

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as f,
            mock.patch.object(self.example_1, "_open_rgb_image", return_value=_FakeImage()),
            mock.patch.object(self.example_1, "_crop_image", return_value=_FakeImage(width=100, height=50)),
        ):
            results = pipeline.extract_tables_from_pdf(f.name, promote_header=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(results[0]["table_on_page"], 1)
        self.assertEqual(results[0]["dataframe"].data, [["A1", "B1"], ["A2", "B2"]])


if __name__ == "__main__":
    unittest.main()
