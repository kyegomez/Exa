import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

class MultiModalInference:
    """
    
    mmi = MultiModalInference()

    user_input = "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
    response = mmi.chat(user_input)
    print(response)

    user_input = "User: And who is that? https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
    response = mmi.chat(user_input)
    print(response)

    mmi.set_checkpoint("new_checkpoint")
    mmi.set_device("cpu")
    mmi.set_max_length(200)
    mmi.clear_chat_history()

    """
    def __init__(
        self,
        checkpoint="HuggingFaceM4/idefics-9b-instruct",
        device=None,
        torch_dtype=torch.bfloat16,
        max_length=100
    ):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype,   
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(checkpoint)

        self.max_length = max_length

        self.chat_history = []
    
    def run(
        self,
        prompts,
        batched_mode=True
    ):
        inputs = self.processor(
            prompts,
            add_end_of_utterance_token=False,
            return_tensors="pt"
        ).to(self.device) if batched_mode else self.processor(
            prompts[0],
            return_tensors="pt"
        ).to(self.device)
        

        exit_condition = self.processor.tokenizer(
            "<end_of_utterance>",
            add_special_tokens=False
        ).input_ids

        bad_words_ids = self.processor.tokenizer(
            [
                "<image>",
                "<fake_token_around_image"
            ],

            add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_length=self.max_length,
        )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        return generated_text
    
    def chat(self, user_input):
        self.chat_history.append(user_input)
        
        prompts = [self.chat_history]
        
        response = self.run(prompts)[0]

        self.chat_history.append(response)

        return response
    
    def set_checkpoint(self, checkpoint):
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
    
    def set_device(self, device):
        self.device = device
        self.model.to(self.device)
    
    def set_max_length(self, max_length):
        self.max_length = max_length
    
    def clear_chat_history(self):
        self.chat_history = []
    




