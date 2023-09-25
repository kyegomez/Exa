# Module/Class Name: MultiModalInference

The `MultiModalInference` class is designed to facilitate multimodal inference using pre-trained models available in the Hugging Face Model Hub. This class allows you to generate text from a combination of textual prompts and images using state-of-the-art vision and language models. It provides flexibility in customizing the inference process, changing device configurations, and managing chat history.

## Key Features

- Multimodal inference: Generate text outputs based on a combination of textual prompts and images.
- Configurable device: Choose the device (CPU or GPU) for inference.
- Model checkpoint selection: Easily switch between different pre-trained model checkpoints.
- Maximum text length: Control the maximum length of generated text.
- Chat history management: Keep track of conversation history for bidirectional interactions.

## Class Definition

```python
class MultiModalInference:
    """
    A class for multimodal inference using pre-trained models from the Hugging Face Hub.

    Attributes
    ----------
    device : str
        The device to use for inference (default is GPU if available, otherwise CPU).
    checkpoint : str, optional
        The name of the pre-trained model checkpoint (default is "HuggingFaceM4/idefics-9b-instruct").
    torch_dtype : torch.dtype, optional
        The torch data type used for model input (default is torch.bfloat16).
    max_length : int
        The maximum length of the generated text (default is 100).
    chat_history : list
        The chat history.

    Methods
    -------
    run(prompts, batched_mode=True)
        Generates text based on the provided prompts.
    chat(user_input)
        Engages in a continuous bidirectional conversation based on the user input.
    set_checkpoint(checkpoint)
        Changes the model checkpoint.
    set_device(device)
        Changes the device used for inference.
    set_max_length(max_length)
        Changes the maximum length of the generated text.
    clear_chat_history()
        Clears the chat history.
    """
    def __init__(
        self,
        checkpoint="HuggingFaceM4/idefics-9b-instruct",
        device=None,
        torch_dtype=torch.bfloat16,
        max_length=100
    ):
        ...
        
    def run(
        self,
        prompts,
        batched_mode=True
    ):
        ...
    
    def chat(self, user_input):
        ...
    
    def set_checkpoint(self, checkpoint):
        ...
    
    def set_device(self, device):
        ...
    
    def set_max_length(self, max_length):
        ...
    
    def clear_chat_history(self):
        ...
```

## Usage

### Initialization and Basic Usage

```python
from exa import MultiModalInference

# Initialize the MultiModalInference instance
mmi = MultiModalInference()

# Define a user input with an image URL
user_input = "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"

# Engage in a chat conversation and generate a response
response = mmi.chat(user_input)

# Print the model's response
print(response)
```

### Modifying Configuration

```python
# Change the model checkpoint
mmi.set_checkpoint("new_checkpoint")

# Change the device to CPU
mmi.set_device("cpu")

# Change the maximum length of generated text
mmi.set_max_length(200)

# Clear the chat history
mmi.clear_chat_history()
```

## Functionality and Usage

### `__init__` Method

The `__init__` method initializes the `MultiModalInference` class.

- `checkpoint` (str, optional): The name of the pre-trained model checkpoint (default is "HuggingFaceM4/idefics-9b-instruct").
- `device` (str, optional): The device to use for inference (default is GPU if available, otherwise CPU).
- `torch_dtype` (torch.dtype, optional): The torch data type used for model input (default is torch.bfloat16).
- `max_length` (int): The maximum length of the generated text (default is 100).

### `run` Method

The `run` method generates text based on the provided prompts.

- `prompts` (list): A list of prompts. Each prompt can include both text strings and images.
- `batched_mode` (bool, optional): Whether to process prompts in batched mode. If True, all prompts are processed together; if False, only the first prompt is processed (default is True).

Returns:
- A list of generated text strings.

### `chat` Method

The `chat` method engages in a continuous bidirectional conversation based on user input.

- `user_input` (str): The user's input text.

Returns:
- The model's response as a string.

### `set_checkpoint` Method

The `set_checkpoint` method allows you to change the model checkpoint.

- `checkpoint` (str): The name of the new pre-trained model checkpoint.

### `set_device` Method

The `set_device` method changes the device used for inference.

- `device` (str): The new device to use for inference (e.g., "cpu" or "cuda").

### `set_max_length` Method

The `set_max_length` method changes the maximum length of the generated text.

- `max_length` (int): The new maximum length for generated text.

### `clear_chat_history` Method

The `clear_chat_history` method clears the chat history, allowing you to start a new conversation.

## Additional Information

- The class handles both textual prompts and images, making it versatile for various multimodal tasks.
- It utilizes the Hugging Face Transformers library for seamless integration with pre-trained models.
- You can easily switch between different model checkpoints, devices, and text length configurations to suit your needs.

For more information and advanced use cases, please refer to the [Hugging Face Transformers documentation](https://huggingface.co/transformers/).

## References

- Hugging Face Transformers Documentation: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- Example Usage: Provided in the Usage section above.

This documentation provides comprehensive information about the `MultiModalInference` class, including its purpose, attributes, methods, and usage examples. It is designed to help you effectively utilize the class for multimodal inference tasks in your projects.