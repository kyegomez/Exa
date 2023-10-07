from exa.utils import Deploy
from exa import Inference

model = Inference(model_id="georgesung/llama2_7b_chat_uncensored", quantize=True)

api = Deploy()
api.load_model()
api.generate("Hello, my name is whaaa")
api.run()
