[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Exa
Boost your GPU's LLM performance by 300% on everyday GPU hardware, as validated by renowned developers, in just 5 minutes of setup and with no additional hardware costs.

-----

## Principles
- Radical Simplicity (Utilizing super-powerful LLMs with as minimal lines of code as possible)
- Ultra-Optimizated Peformance (High Performance code that extract all the power from these LLMs)
- Fludity & Shapelessness (Plug in and play and re-architecture as you please)

---

# 🤝 Schedule a 1-on-1 Session
Book a [1-on-1 Session with Kye](https://calendly.com/apacai/agora), the Creator, to discuss any issues, provide feedback, or explore how we can improve Exa for you.

---

## 📦 Installation 📦
You can install the package using pip

```bash
pip install exxa
```
-----



# Usage

## Inference
Generate text using pretrained models with optional quantization with minimal configuration and straightforward usage.

- Load specified pre-trained models with device flexibility (CPU/CUDA).
- Set a default maximum length for the generated sequences.
- Choose to quantize model weights for faster inference.
- Use a custom configuration for quantization as needed.
- Generate text through either a direct call or the run method.
- Simple usage for quick text generation based on provided prompts.

```python
from exa import Inference

model = Inference(
    model_id="georgesung/llama2_7b_chat_uncensored",
    quantize=True
)

model.run("What is your name")
```
---

## GPTQ Inference
Efficiently generate text using quantized GPT-like models built for HuggingFace's pre-trained models with optional quantization and only a few lines of code for instantiation and generation.

- Load specified pre-trained models with an option for quantization.
- Define custom bit depth for the quantization (default is 4 bits).
- Fine-tune quantization parameters using specific datasets.
- Set maximum length for generated sequences to maintain consistency.
- Tokenize prompts and generate text based on them seamlessly.

```python
# !pip install exxa
from exa import GPTQInference

model_id = "gpt2-medium"
inference = GPTQInference(
    model_id, 
    quantization_config_bits=2, 
    max_length=400, 
    quantization_config_dataset='c4'
)
output_text = inference.run("The future of AI is")
print(output_text)
```
---

## Quantize
Achieve smaller model sizes and faster inference by utilizing a unified interface tailored to HuggingFace's framework and only a simple class instantiation with multiple parameters is needed.
- Efficiently quantize HuggingFace's pretrained models with specified bits (default is 4 bits).
- Set custom thresholds for quantization for precision management.
- Ability to skip specific modules during quantization for sensitive model parts.
- Offload parts of the model to CPU in FP32 format for GPU memory management.
- Specify if model weights are already in FP16 format.
- Choose from multiple quantization types like "fp4", "int8", and more.
- Option to enable double quantization for more compression.
- Verbose logging for a detailed understanding of the quantization process.
- Seamlessly push to and load models from the HuggingFace model hub.
- In-built logger initialization tailored for quantization logs.
- Log metadata for state and settings introspection.


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

[Check out our project board for our current backlog and features we're implementing](https://github.com/users/kyegomez/projects/8/views/2)

------

# Benchmarks
The following is what we benchmark for according to the [🤗 LLM-Perf Leaderboard 🏋️ benchmarks](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)


- https://huggingface.co/spaces/optimum/llm-perf-leaderboard
- https://github.com/huggingface/optimum-benchmark

## Metrics 
- Backend 🏭
- Dtype 📥
- Optimizations 🛠️
- Quantization 🗜️
- Class 🏋️
- Type 🤗
- Memory (MB) ⬇️
- Throughput (tokens/s) ⬆️
- Energy (tokens/kWh) ⬇️
- Best Score (%) ⬆️
- Best Scored LLM 🏆

# License
MIT

# Todo

- Setup utils logger classes for metric logging with useful metadata such as token inference per second, latency, memory consumption
- Add cuda c++ extensions for radically optimized classes for high performance quantization + inference on the edge



