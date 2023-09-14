import logging

import torch
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTQInference:
    def __init__(
        self,
        model_id,
        quantization_config_bits,
        quantization_config_dataset,
        max_length,
        verbose = False,
        distributed = False,
    ):
        self.model_id = model_id
        self.quantization_config_bits = quantization_config_bits
        self.quantization_config_dataset = quantization_config_dataset
        self.max_length = max_length
        self.verbose = verbose
        self.distributed = distributed

        if self.distributed:
            assert torch.cuda.device_count() > 1, "You need more than 1 gpu for distributed processing"
            set_start_method("spawn", force=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.quantization_config = GPTQConfig(
            bits=self.quantization_config_bits,
            dataset=quantization_config_dataset,
            tokenizer=self.tokenizer
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.quantization_config
        ).to(self.device)

        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[0],
                output_device=0,
            )
        
        logger.info(f"Model loaded from {self.model_id} on {self.device}")

    def run(
            self, 
            prompt: str,
            max_length: int = 500,
        ):
        max_length = self.max_length or max_length
        
        try:
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    do_sample=True
                )

            return self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
        
        except Exception as error:
            print(f"Error: {error} in inference mode, please change the inference logic or try again")
            raise
    
    def __del__(self):
        #free up resources
        torch.cuda.empty_cache()

        