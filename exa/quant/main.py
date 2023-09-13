from transformers import AutoModelForCasualLM, AutoTokenizer, BitsAndBytesConfig

class Quantize:

    #TODO: Add documentation as docstrings and in the docs folder
    #TODO: Add extensive error handling
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
    ):
        super().__init__()
        self.model_id

        self.bits
        self.threshold = threshold
        self.skip_modules = skip_modules
        self.enable_fp32_cpu_offload = enable_fp32_cpu_offload
        self.has_fp16_weight = has_fp16_weight
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.use_double_quant = use_double_quant

        self.tokenizer = None
        self.model = None

    def load_model(self):
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
            bnb_4bit_use_double_quant=self.use_doubkle_quant,   
        )

        #load the quantized model
        self.model = AutoModelForCasualLM.from_pretrained(
            self.model_id,
            device_map=device_map,
            quantization_config=quantization_config,
        )
    
    def push_to_hub(self, hub):
        #push the quantized model to the hub
        if self.model is not None and self.tokenizer is not None:
            self.model.push_to_hub(hub)
            self.tokenizer.push_to_hub(hub)
        else:
            raise ValueError("Model and tokenizer must be loaded before pushing to the hub")
    
    def load_from_hub(self, hub):
        #load a quantized model from the hub
        if self.tokenizer is not None:
            self.model = AutoModelForCasualLM.from_pretrained(hub, device_map="auto")
        else:
            raise ValueError("Tokenizer must be loaded from loading the hub model")

#usage
# quantize = Quantize(
#     model_id="bigscience/bloom-1b7",
#     bits=8,
#     enable_fp32_cpu_offload=True,
# )

# quantize.load_model()
# quantize.push_to_hub("my model")
# quantize.load_from_hub('my model')
