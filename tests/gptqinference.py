import unittest
import torch
from unittest.mock import patch, Mock
from transformers import PreTrainedTokenizerFast, PreTrainedModel
from exa.inference.gptq import GPTQInference

class TestGPTQInference(unittest.TestCase):

    def setUp(self):
        # Mocking some of the external dependencies to avoid actual calls
        self.mocked_tokenizer = Mock(spec=PreTrainedTokenizerFast)
        self.mocked_tokenizer.encode.return_value = torch.tensor([0, 1, 2])
        self.mocked_tokenizer.decode.return_value = "decoded text"
        
        self.mocked_model = Mock(spec=PreTrainedModel)
        self.mocked_model.generate.return_value = torch.tensor([[0, 1, 2]])

        patcher1 = patch('YOUR_MODULE_PATH.AutoTokenizer.from_pretrained', return_value=self.mocked_tokenizer)
        patcher2 = patch('YOUR_MODULE_PATH.AutoModelForCausalLM.from_pretrained', return_value=self.mocked_model)

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        
        self.mocked_from_pretrained_tokenizer = patcher1.start()
        self.mocked_from_pretrained_model = patcher2.start()

    def test_initialization(self):
        inferer = GPTQInference(
            model_id='gpt-2', 
            quantization_config_bits=8, 
            quantization_config_dataset='wiki', 
            max_length=100
        )
        self.assertTrue(inferer.verbose == False)
        self.assertTrue(inferer.distributed == False)
        self.mocked_from_pretrained_tokenizer.assert_called_once()
        self.mocked_from_pretrained_model.assert_called_once()

    def test_initialization_with_distributed(self):
        with self.assertRaises(AssertionError):
            GPTQInference(
                model_id='gpt-2', 
                quantization_config_bits=8, 
                quantization_config_dataset='wiki', 
                max_length=100,
                distributed=True
            )

    @patch('torch.cuda.device_count', return_value=2)
    def test_initialization_distributed_success(self, mocked_device_count):
        inferer = GPTQInference(
            model_id='gpt-2', 
            quantization_config_bits=8, 
            quantization_config_dataset='wiki', 
            max_length=100,
            distributed=True
        )
        self.assertTrue(inferer.distributed == True)

    def test_run(self):
        inferer = GPTQInference(
            model_id='gpt-2', 
            quantization_config_bits=8, 
            quantization_config_dataset='wiki', 
            max_length=100
        )
        result = inferer.run("This is a test.")
        self.assertEqual(result, "decoded text")
        
        self.mocked_tokenizer.encode.assert_called_once_with("This is a test.", return_tensors="pt")
        self.mocked_model.generate.assert_called_once()

    def test_run_error(self):
        self.mocked_tokenizer.encode.side_effect = Exception("Tokenization failed")
        inferer = GPTQInference(
            model_id='gpt-2', 
            quantization_config_bits=8, 
            quantization_config_dataset='wiki', 
            max_length=100
        )
        with self.assertRaises(Exception) as context:
            inferer.run("This is a test.")
        self.assertEqual(str(context.exception), "Tokenization failed")

    def test_del(self):
        inferer = GPTQInference(
            model_id='gpt-2', 
            quantization_config_bits=8, 
            quantization_config_dataset='wiki', 
            max_length=100
        )
        with patch('torch.cuda.empty_cache') as mocked_empty_cache:
            del inferer
            mocked_empty_cache.assert_called_once()

if __name__ == '__main__':
    unittest.main()
