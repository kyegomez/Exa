import logging
import time
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class Quantize:
    """
    Quantize provides a convenient way to load, quantize, and manage HuggingFace models, specifically designed for optimization.
    
    The primary goal of this class is to help users quantize HuggingFace models using the BitsAndBytes configuration to achieve faster inference times and to either push the optimized models to HuggingFace's model hub or load them from it.

    Parameters:
    - model_id (str): Model identifier to be loaded from the HuggingFace hub.
    - bits (int, optional): Bit precision for quantization. Default is 4.
    - threshold (float, optional): Threshold for quantization. Default is 6.0.
    - skip_modules (list, optional): List of modules to skip during quantization. Default is None.
    - enable_fp32_cpu_offload (bool, optional): If True, offload the FP32 computations to CPU. Default is False.
    - has_fp16_weight (bool, optional): If True, indicates that the model has FP16 weights. Default is False.
    - compute_dtype (str, optional): Data type for computation after quantization. Default is None.
    - quant_type (str, optional): Type of quantization to apply. Default is "fp4".
    - use_double_quant (bool, optional): If True, applies double quantization. Default is False.
    - verbose (bool, optional): If True, provides more detailed logs. Default is False.
    """
    def __init__(
        self,
        model_id,
        bits: int = 4,
        threshold: float = 6.0,
        skip_modules = None,
        enable_fp32_cpu_offload=False,
        has_fp16_weight=False,
        compute_dtype=None,
        quant_type: str = "fp4",
        use_double_quant=False,
        verbose=False,
    ):
        super().__init__()
        self.model_id = model_id

        self.bits = bits
        self.threshold = threshold
        self.skip_modules = skip_modules
        self.enable_fp32_cpu_offload = enable_fp32_cpu_offload
        self.has_fp16_weight = has_fp16_weight
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.use_double_quant = use_double_quant

        self.verbose = verbose

        self.tokenizer = None
        self.model = None
        self.logger = self._init_logger()
    
    def _init_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if self.verbose else logging.ERROR)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    
    def log_metadata(self, metadata):
        if self.verbose:
            for key, value in metadata.items():
                self.logger.info(f"{key}: {value}")


    def load_model(self):
        """
        Loads and quantizes the HuggingFace model using the provided parameters and configuration.
        """
        try:

            #load the tokenizer and model

            #define device map if cpu offload is enabled
            device_map = None
            if self.enable_fp32_cpu_offload:
                device_map = {
                    "transformer.word_embeddings": 0,
                    "transformer.word_embeddings_layernorm": 0,
                    "lm_head": "cpu",
                    "transformer.h": 0,
                    "transformer.ln_f": 0,
                }
            
            #quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_8bit = self.bits == 8,
                load_in_4bit = self.bits == 4,
                llm_int8_threshold=self.threshold,
                llm_int8_skip_modules=self.skip_modules,
                llm_int8_enable_fp32_cpu_offload=self.enable_fp32_cpu_offload,
                llm_int8_has_fp16_weight = self.has_fp16_weight,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_quant_type=self.quant_type,
                bnb_4bit_use_double_quant=self.use_double_quant,   
            )

            start_time = time.time()

            #load the quantized model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                quantization_config=quantization_config,
            )

            #end timer
            end_time = time.time()
            #log metadata and metrics
            metadata = {
                "Bits": self.bits,
                "Threshold": self.threshold,
                "FP32 CPU Offload": self.enable_fp32_cpu_offload,
                "Quantization Type": self.quant_type,
                "Time to Quantize (s)": round(end_time - start_time, 2)
            }

            self.log_metadata(metadata)
        except RuntimeError as error:
            self.logger.error(f"An error occured while loading the model: {error} please understand the root cause then look at documentation for common errors")
    
    def push_to_hub(self, hub):
        """
        Pushes the quantized model and the associated tokenizer to the HuggingFace's model hub.

        Parameters:
        - hub (str): Hub name or identifier to which the model and tokenizer should be pushed.
        """
        try:
                
            #push the quantized model to the hub
            if self.model is not None and self.tokenizer is not None:
                self.model.push_to_hub(hub)
                self.tokenizer.push_to_hub(hub)
            else:
                raise ValueError("Model and tokenizer must be loaded before pushing to the hub")
        except RuntimeError as error:
            self.logger.error(f"An error occured while pushing to the hub: {error}")
    
    def load_from_hub(self, hub):
        """
        Loads a quantized model from the HuggingFace's model hub.

        Parameters:
        - hub (str): Hub name or identifier from which the model should be loaded.
        """
        try:

            #load a quantized model from the hub
            if self.tokenizer is not None:
                self.model = AutoModelForCausalLM.from_pretrained(hub, device_map="auto")
            else:
                raise ValueError("Tokenizer must be loaded from loading the hub model")
        except RuntimeError as error:
            self.logger.error(f'An error occured while loading from the hub: {error}')


#usage
# quantize = Quantize(
#     model_id="bigscience/bloom-1b7",
#     bits=8,
#     enable_fp32_cpu_offload=True,
# )

# quantize.load_model()
# quantize.push_to_hub("my model")
# quantize.load_from_hub('my model')
