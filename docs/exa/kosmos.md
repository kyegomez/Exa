# Kosmos Documentation

Kosmos is a model to perform multimodal tasks by combining natural language understanding and computer vision capabilities. It leverages a pre-trained model to enable tasks such as multimodal grounding, referring expression comprehension, referring expression generation, grounded visual question answering (VQA), and grounded image captioning.

## Class: Kosmos

### Purpose

The `Kosmos` class is the central component of the Shapeless library. It enables users to interact with pre-trained models for multimodal tasks that involve text and images. By initializing an instance of `Kosmos`, you gain access to various multimodal capabilities.

### Class Definition

```python
class Kosmos:
    def __init__(self, model_name="ydshieh/kosmos-2-patch14-224"):
        """
        Initialize the Kosmos class.

        Parameters:
        - model_name (str): The name of the pre-trained model to use (default: "ydshieh/kosmos-2-patch14-224").
        """
```

### How It Works

The `Kosmos` class utilizes a pre-trained model to perform multimodal tasks. It takes text and image inputs and returns results based on the model's understanding of the multimodal context.

### Usage

#### 1. Multimodal Grounding

```python
kosmos = Kosmos()
kosmos.multimodal_grounding("Find the red apple in the image.", "https://example.com/apple.jpg")
```

#### 2. Referring Expression Comprehension

```python
kosmos.referring_expression_comprehension("Show me the green bottle.", "https://example.com/bottle.jpg")
```

#### 3. Referring Expression Generation

```python
kosmos.referring_expression_generation("It is on the table.", "https://example.com/table.jpg")
```

#### 4. Grounded Visual Question Answering (VQA)

```python
kosmos.grounded_vqa("What is the color of the car?", "https://example.com/car.jpg")
```

#### 5. Grounded Image Captioning

```python
kosmos.grounded_image_captioning("https://example.com/beach.jpg")
```

### Additional Tips

- You can use `Kosmos` for various multimodal tasks by calling the respective functions.
- You can specify a different pre-trained model by providing the `model_name` parameter during initialization.
- Use the `show` and `save_path` parameters when visualizing results to control display and save images.

## Function: `draw_entity_boxes_on_image`

### Purpose

The `draw_entity_boxes_on_image` function allows you to draw bounding boxes around entities in an image. It enhances the visual representation of detected entities.

### Function Definition

```python
def draw_entity_boxes_on_image(image, entities, show=False, save_path=None):
    """
    Draw bounding boxes around entities in an image.

    Parameters:
    - image (str, Image.Image, torch.Tensor): The input image in various formats (image path, PIL image, or torch.Tensor).
    - entities (list): A list of entities, where each entity is a tuple containing entity name, position (start, end), and bounding boxes.
    - show (bool): If True, display the image with bounding boxes (default: False).
    - save_path (str): If specified, save the image with bounding boxes to the given path.

    Returns:
    - np.ndarray: The image with bounding boxes.
    """
```

### How It Works

The `draw_entity_boxes_on_image` function takes an image and a list of entities as input. It then draws bounding boxes around the specified entities in the image.

### Usage Example

```python
entities = [("red apple", (0, 1), [(0.2, 0.3, 0.4, 0.5)]), ("green bottle", (2, 3), [(0.6, 0.7, 0.8, 0.9)])]
image_path = "image.jpg"

result_image = draw_entity_boxes_on_image(image_path, entities, show=True, save_path="output.jpg")
```

### Additional Tips

- The `show` parameter controls whether the image with bounding boxes is displayed.
- You can use the `save_path` parameter to save the annotated image to a file.

## Function: `generate_boxes`

### Purpose

The `generate_boxes` function allows you to generate bounding boxes around entities in an image based on a given prompt. It simplifies the process of highlighting entities in an image.

### Function Definition

```python
def generate_boxes(prompt, image_url):
    """
    Generate bounding boxes around entities in an image based on a prompt.

    Parameters:
    - prompt (str): The prompt describing the entities to be highlighted.
    - image_url (str): The URL of the image to process.

    Returns:
    - np.ndarray: The image with generated bounding boxes.
    """
```

### How It Works

The `generate_boxes` function takes a prompt and an image URL as input. It processes the prompt and generates bounding boxes around the entities mentioned in the prompt within the given image.

### Usage Example

```python
prompt = "Find the red apple in the image."
image_url = "https://example.com/apple.jpg"

result_image = generate_boxes(prompt, image_url)
```

### Additional Tips

- You can use this function to quickly visualize entities mentioned in a prompt within an image.
- Ensure that the image URL is accessible and contains the image you want to process.

## Conclusion

The Shapeless library simplifies multimodal tasks involving natural language and computer vision. By utilizing the `Kosmos` class and related functions, you can easily perform tasks such as multimodal grounding, referring expression comprehension, referring expression generation, grounded VQA, and grounded image captioning. Additionally, the ability to draw bounding boxes and generate boxes enhances the visual representation of entities in images, making it a powerful tool for various applications.
