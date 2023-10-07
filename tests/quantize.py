import unittest
from unittest.mock import MagicMock, patch
from transformers import AutoModelForCausalLM
from exa.quant.main import Quantize


class TestQuantize(unittest.TestCase):
    def setUp(self):
        self.quantize = Quantize(
            model_id="bigscience/bloom-1b7",
            bits=8,
            enable_fp32_cpu_offload=True,
        )

    @patch.object(AutoModelForCausalLM, "from_pretrained")
    def test_load_model(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        self.quantize.load_model()

        mock_from_pretrained.assert_called_once()
        self.assertEqual(self.quantize.model, mock_model)

    @patch.object(AutoModelForCausalLM, "from_pretrained")
    def test_load_model_error(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = RuntimeError("Test error")

        with self.assertRaises(RuntimeError):
            self.quantize.load_model()

    @patch.object(AutoModelForCausalLM, "push_to_hub")
    def test_push_to_hub(self, mock_push_to_hub):
        mock_model = MagicMock()
        self.quantize.model = mock_model

        self.quantize.push_to_hub("test_hub")

        mock_push_to_hub.assert_called_once_with("test_hub")

    @patch.object(AutoModelForCausalLM, "push_to_hub")
    def test_push_to_hub_error(self, mock_push_to_hub):
        mock_push_to_hub.side_effect = RuntimeError("Test error")

        with self.assertRaises(RuntimeError):
            self.quantize.push_to_hub("test_hub")

    @patch.object(AutoModelForCausalLM, "from_pretrained")
    def test_load_from_hub(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        self.quantize.load_from_hub("test_hub")

        mock_from_pretrained.assert_called_once_with("test_hub", device_map="auto")
        self.assertEqual(self.quantize.model, mock_model)

    @patch.object(AutoModelForCausalLM, "from_pretrained")
    def test_load_from_hub_error(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = RuntimeError("Test error")

        with self.assertRaises(RuntimeError):
            self.quantize.load_from_hub("test_hub")

    def test_init_logger(self):
        logger = self.quantize._init_logger()
        self.assertEqual(logger.level, 40)  # 40 is the level for ERROR

    def test_log_metadata(self):
        with patch("logging.Logger.info") as mock_info:
            self.quantize.verbose = True
            self.quantize.log_metadata({"test": "value"})
            mock_info.assert_called_once_with("test: value")


if __name__ == "__main__":
    unittest.main()
