[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Exa
Ultra-optimized fast inference library for running exascale LLMs locally on modern consumer-class GPUs.

## Principles
- Radical Simplicity (Utilizing super-powerful LLMs with as minimal code as possible)
- Ultra-Optimizated (High Performance classes that extract all the power from these LLMs)
- Fludity & Shapelessness (Plug in and play and re-architecture as you please)

---

# 🤝 Schedule a 1-on-1 Session
Book a [1-on-1 Session with Kye](https://calendly.com/apacai/agora), the Creator, to discuss any issues, provide feedback, or explore how we can improve Zeta for you.

---

## 📦 Installation 📦
You can install the package using pip

```bash
pip install exxa
```
-----



# Usage

## Inference
```python
from exa import Inference

model = Inference(
    model_id="georgesung/llama2_7b_chat_uncensored",
    quantized=True
)

model.run("What is your name")
```


## GPTQ Inference

```python

from exa import GPTQInference

model_id = "facebook/opt-125m"
model = GPTQInference(model_id=model_id, max_length=400)

prompt = "in a land far far away"
result = model.run(prompt)
print(result)

```

## Quantize

```python
from exa import Quantize

#usage
quantize = Quantize(
     model_id="bigscience/bloom-1b7",
     bits=8,
     enable_fp32_cpu_offload=True,
)

quantize.load_model()
quantize.push_to_hub("my model")
quantize.load_from_hub('my model')


```

-----

## 🎉 Features 🎉

- **World-Class Quantization**: Get the most out of your models with top-tier performance and preserved accuracy! 🏋️‍♂️
  
- **Automated PEFT**: Simplify your workflow! Let our toolkit handle the optimizations. 🛠️

- **LoRA Configuration**: Dive into the potential of flexible LoRA configurations, a game-changer for performance! 🌌

- **Seamless Integration**: Designed to work seamlessly with popular models like LLAMA, Falcon, and more! 🤖

----

## 💌 Feedback & Contributions 💌

We're excited about the journey ahead and would love to have you with us! For feedback, suggestions, or contributions, feel free to open an issue or a pull request. Let's shape the future of fine-tuning together! 🌱

------


# License
MIT



