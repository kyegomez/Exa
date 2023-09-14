from exa import Inference

model = Inference(
    model_id="georgesung/llama2_7b_chat_uncensored",
    quantize=True
)

model.run("What is your name")