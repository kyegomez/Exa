import unittest
from unittest.mock import Mock, patch

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from exa.inference.hf import Inference


class TestInference(unittest.TestCase):
    def setUp(self):
        self.mocked_tokenizer = Mock(spec=PreTrainedTokenizerFast)
        self.mocked_tokenizer.encode.return_value = torch.tensor([0, 1, 2])
        self.mocked_tokenizer.decode.return_value = "generated text"

        self.mocked_model = Mock(spec=PreTrainedModel)
        self.mocked_model.generate.return_value = torch.tensor([[0, 1, 2]])

        self.tokenizer_patcher = patch(
            "YOUR_MODULE_PATH.AutoTokenizer.from_pretrained",
            return_value=self.mocked_tokenizer,
        )
        self.model_patcher = patch(
            "YOUR_MODULE_PATH.AutoModelForCausalLM.from_pretrained",
            return_value=self.mocked_model,
        )

        self.mocked_from_pretrained_tokenizer = self.tokenizer_patcher.start()
        self.mocked_from_pretrained_model = self.model_patcher.start()

    def tearDown(self):
        self.tokenizer_patcher.stop()
        self.model_patcher.stop()

    def test_default_initialization(self):
        inference = Inference("gpt-2")
        self.assertEqual(inference.model_id, "gpt-2")
        self.assertEqual(inference.max_length, 20)
        self.assertEqual(inference.verbose, False)
        self.assertEqual(inference.distributed, False)
        self.assertEqual(inference.decoding, False)

    def test_model_loading(self):
        inference = Inference("gpt-2")
        inference.load_model()
        self.mocked_from_pretrained_tokenizer.assert_called_once()
        self.mocked_from_pretrained_model.assert_called_once()

    def test_gpu_device_assignment(self):
        with patch("torch.cuda.is_available", return_value=True):
            inference = Inference("gpt-2")
            self.assertEqual(inference.device, "cuda")

    def test_default_quantization(self):
        inference = Inference("gpt-2", quantize=True)
        self.assertIsNotNone(inference.quantization_config)
        self.assertTrue(inference.quantization_config["load_in_4bit"])
        self.assertEqual(
            inference.quantization_config["bnb_4bit_compute_dtype"], torch.bfloat16
        )

    def test_custom_quantization(self):
        custom_config = {"load_in_4bit": False, "bnb_4bit_use_double_quant": False}
        inference = Inference("gpt-2", quantize=True, quantization_config=custom_config)
        self.assertFalse(inference.quantization_config["load_in_4bit"])
        self.assertFalse(inference.quantization_config["bnb_4bit_use_double_quant"])

    def test_text_generation_without_realtime_decoding(self):
        inference = Inference("gpt-2", decoding=False)
        result = inference.run("This is a test.")
        self.assertEqual(result, "generated text")

    def test_text_generation_with_realtime_decoding(self):
        inference = Inference("gpt-2", decoding=True)
        with patch("builtins.print") as mocked_print:
            result = inference.run("This is a test.")
        mocked_print.assert_called()
        self.assertEqual(result, "generated text")

    def test_distributed_processing_assertion(self):
        with patch("torch.cuda.device_count", return_value=1):
            with self.assertRaises(AssertionError):
                Inference("gpt-2", distributed=True)

    def test_error_during_model_loading(self):
        self.mocked_from_pretrained_tokenizer.side_effect = Exception(
            "Failed to load tokenizer"
        )
        with self.assertRaises(Exception):
            Inference("gpt-2")

    def test_text_generation_failure_handling(self):
        self.mocked_model.generate.side_effect = Exception("Failed to generate")
        inference = Inference("gpt-2")
        with self.assertRaises(Exception):
            inference.run("This is a test.")


if __name__ == "__main__":
    unittest.main()
