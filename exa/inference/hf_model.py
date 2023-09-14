import logging

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from exa.utils.custom_formatter import ColoredLogger


class Inference:
    """
    A class for running inference on a given model.

    Attributes:
        model_id (str): The ID of the model.
        device (str): The device to run the model on (either 'cuda' or 'cpu').
        max_length (int): The maximum length of the output sequence.
    """
    def __init__(
            self, 
            model_id: str, 
            device: str = None, 
            max_length: int = 20, 
            quantize: bool = False, 
            quantization_config: dict = None,
            verbose = False,
            logger=None,
            distributed=False,
        ):
        """
        Initialize the Inference object.
        
        Args:
            model_id (str): The ID of the model.
            device (str, optional): The device to run the model on. Defaults to 'cuda' if available.
            max_length (int, optional): The maximum length of the output sequence. Defaults to 20.
            quantize (bool, optional): Whether to use quantization. Defaults to False.
            quantization_config (dict, optional): The configuration for quantization.
            verbose (bool, optional): Whether to print verbose logs. Defaults to False.
            logger (logging.Logger, optional): The logger to use. Defaults to a basic logger.
        """
        self.logger = logging.getLogger(__name__)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.max_length = max_length
        self.verbose = verbose
        self.model, self.tokenizer = None, None

        if self.distributed:
            assert torch.cuda.device_count() > 1, "You need more than 1 gpu for distributed processing"


        bnb_config = None
        if quantize:
            if not quantization_config:
                quantization_config = {
                    'load_in_4bit': True,
                    'bnb_4bit_use_double_quant': True,
                    'bnb_4bit_quant_type': "nf4",
                    'bnb_4bit_compute_dtype': torch.bfloat16
                }
            bnb_config = BitsAndBytesConfig(**quantization_config)

        # try:
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         self.model_id, 
        #         quantization_config=bnb_config
        #     )
            
        #     self.model.to(self.device)
        # except Exception as e:
        #     self.logger.error(f"Failed to load the model or the tokenizer: {e}")
        #     raise

    def load_model(self):
        if not self.model or not self.tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                bnb_config = BitsAndBytesConfig(**self.quantization_config) if self.quantization_config else None
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config
                ).to(self.device)
                
                if self.distributed:
                    self.model = DDP(self.model)
            except Exception as error:
                self.logger.error(f"Failed to load the model or the tokenizer: {error}")
                raise

    def __call__(
            self, 
            prompt_text: str, 
            max_length: int = None
        ):

        max_length = max_length if max_length else self.max_length
        try:
            inputs = self.tokenizer.encode(
                prompt_text, 
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise


    def run(
            self, 
            prompt_text: str, 
            max_length: int = None
        ):
        """
        Generate a response based on the prompt text.

        Args:
        - prompt_text (str): Text to prompt the model.
        - max_length (int): Maximum length of the response.

        Returns:
        - Generated text (str).
        """
        self.load_model()

        max_length = max_length if max_length else self.max_length
        try:
            inputs = self.tokenizer.encode(
                prompt_text, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    max_length=max_length, 
                    do_sample=True
                )
            
            del inputs

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise
