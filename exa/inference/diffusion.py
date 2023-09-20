import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import PartialState
from diffusers import DiffusionPipeline


class Diffuse:
    def __init__(
        self,
        model,
        dtype,
        use_safetensors=False,
        method="accelerate"
    ):
        self.pipeline = DiffusionPipeline(
            model,
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
        )

        if self.method == "accelerate":
            self.distributed_state = PartialState()
            self.pipeline.to(self.distributed_state.device)
        elif self.method == "torch":
            pass

    def infer_with_accelerate(self, prompt):
        with self.distributed_state.split_between_processes(prompt) as prompt:
            result = self.pipeline(prompt).images[0]
            result.save(f"result_{self.distributed_state.process_index}.png")
    
    def infer_torch(self, rank, world_size, prompts):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        self.pipeline.to(rank)

        prompt = prompts[rank]
        image = self.pipeline(prompt).images[0]
        
        try:
            image.save(f"./{'_'.join(prompt.split())}.png")
        except RuntimeError as error:
            print(f"Error trying to save the image {error}")
    
    def run(self, prompts, world_size=2):
        try:
            if self.method == "accelerate":
                self.infer_with_accelerate(prompts)
        except RuntimeError as error:
            print(f"Could not run inference om accelerate setup: {error}")

        #error handling
        try:
            if self.method == "torch":
                mp.spawn(
                    self.infer_torch, 
                    args=(world_size, prompts),
                    nproces=world_size,
                    join=True
                )
        except RuntimeError as error:
            print(f"Could not run inference on torch setup: {error}")
