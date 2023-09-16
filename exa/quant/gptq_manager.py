import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQuantizer, load_quantized_model
from accelerate import init_empty_weights

class GPTQManager:
    """
    usage

    manager = GPTQManager("facebook/opt-125m")

    manager.quantize()
    manager.save_quantized_model("/path/to/save")
    manager.load_quantized_model("/path/to/save")
    """
    def __init__(
        self,
        model_name,
        bits,
        dataset="c4",
        block_name_to_quantize="model.decoder.layers",
        model_seqlen=2048
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.bits = bits
        self.dataset = dataset
        self.block_name_to_quantize = block_name_to_quantize
    
    @staticmethod
    def quantize(self, save_path):
        """Save"""
        quantizer = GPTQuantizer(
            bits=self.bits,
            dataset=self.dataset,
            block_name_to_quantize=self.block_name_to_quanze,
            model_seqlen=self.model_seqlen
        )
        quantizer.quantize(self.model, save_path)
    
    def load_quantized_model(
        self,
        save_path,
        device_map="auto",
        disable_exllama=False
    ):
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
            empty_model.tie_weights()
            self.quantized_model = load_quantized_model(
                empty_model,
                save_folder=save_path,
                device_map=device_map,
                disable_exllama=disable_exllama
            )
            return self.quantized_model
        