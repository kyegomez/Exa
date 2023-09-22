# Kosmos Inference Class
import os

import cv2
import numpy as np
import requests
import torch
import torchvision.transform as T
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


#utils
def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)





class Kosmos:
    def __init__(
        self,
        model_name="ydshieh/kosmos-2-patch14-224",
    ):
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def get_image(self, url):
        return Image.open(requests.get(url, stream=True).raw)
    
    def run(self, prompt, image):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"][:, :-1],
            attention_mask=inputs["attention_mask"][:, :-1],
            img_features=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1],
            use_cache=True,
            max_new_tokens=64,
        )
        generated_texts = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,   
        )[0]
        processed_text, entities = self.processor.post_process_generation(
            generated_texts
        )
    
    def run(self, prompt, image_url):
        image = self.get_image(image_url)
        processed_text, entities = self.process_pormpt(prompt, image)
        print(processed_text)
        print(entities)
    
    #tasks
    def multimodal_grounding(self, phrase, image_url):
        prompt = f"<grounding><phrase> {phrase} </phrase>"
        self.run(prompt, image_url)
    
    def referring_expression_comprehension(self, phrase, image_url):
        prompt = f"<grounding><phrase> {phrase} </phrase>"
        self.run(prompt, image_url)
    
    def referring_expression_generation(self, phrase, image_url):
        prompt = f"<grounding><phrase> It</phrase><object><patch_index_0044><patch_index_0863></object> is"
        self.run(prompt, image_url)
    
    def grounded_vqa(self, question, image_url):
        prompt = f"<grounding> Question: {question} Answer:"
        self.run(prompt, image_url)
    
    def grounded_image_captioning(self, image_url):
        prompt = f"<grounding> An image of"
        self.run(prompt, image_url)
    
    def grounded_image_captioning_detailed(self, image_url):
        prompt = f"<grounding> Describe this image in detail"
        self.run(prompt, image_url)
    
    def draw_entity_boxes_on_image(
        image,
        entities,
        show=False,
        save_path=None
    ):
        if isinstance(image, Image.Image):
            image_h = image.height
            image_w = image.width
            image = np.array(image)[:, :, [2, 1, 0]]
        
        elif isinstance(image, str):
            if os.path.exists(image):
                phil_img = Image.open(image).convert("RGB")
                image = np.array(phil_img)[:, :, [2, 1, 0]]
                image_h = phil_img.height
                image_w = phil_img.width
            else:
                raise ValueError(f"Invalid image path, {image}")
        elif isinstance(image, torch.Tensor):
            image_tensor = image.cpu()
            reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
            reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
            image_tensor = image_tensor  * reverse_norm_std + reverse_norm_mean
            phil_img = T.ToPILImage()(image_tensor)
            image_h = phil_img.height
            image_w = phil_img.width
            image = np.array(phil_img)[:, :, [2, 1, 0]]
        else:
            raise ValueError(f"Invalid image format, {type(image)} for {image}")
    
        if len(entities) == 0:
            return Image
        
        new_image = image.copy()

        previous_boxes  = []
        text_size = 1
        text_line = 1
        box_line = 3

        
