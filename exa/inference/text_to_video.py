import logging

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
from termcolor import colored


class TextToVideo:
    """
    Text-to-Video class for generating high-quality videos from text prompts.

    Args:
        model_name (str): The name of the pre-trained video model.
        prompt (str): The text prompt for generating the video.
        num_inference_steps (int): The number of inference steps.
        height (int): The height of the video frames.
        width (int): The width of the video frames.
        num_frames (int): The number of frames in the video.
        strength (float): The denoise strength for upscaling the video frames.
        output_video_path (str): The path to save the generated video.

    Attributes:
        model_name (str): The name of the pre-trained video model.
        prompt (str): The text prompt for generating the video.
        num_inference_steps (int): The number of inference steps.

        height (int): The height of the video frames.
        width (int): The width of the video frames.
        num_frames (int): The number of frames in the video.

        strength (float): The denoise strength for upscaling the video frames.
        output_video_path (str): The path to save the generated video.

    #Usage:
    ```
    # Import the class
    from exa import TextToVideo

    # Define the parameters
    model_name = 'cerspense/zeroscope_v2_XL'
    prompt = 'A beautiful sunset over the ocean'
    num_inference_steps = 100
    height = 720
    width = 1280
    num_frames = 30
    strength = 0.8
    output_video_path = '/path/to/save/video.mp4'

    # Create an instance of the class
    text_to_video = TextToVideo(
        model_name=model_name,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        strength=strength,
        output_video_path=output_video_path
    )

    # Generate the video
    video_path = text_to_video.run()

    # Print the path to the generated video
    print(f'The video was saved at: {video_path}')
    ```
    """

    def __init__(
        self,
        model_name,
        prompt,
        num_inference_steps,
        height,
        width,
        num_frames,
        strength,
        output_video_path
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps

        self.height = height
        self.width = width
        self.num_frames = num_frames

        self.strength = strength
        self.output_video_path = output_video_path

    def run(self):
        """
        Generate a video from the text prompt.

        Returns:
            str: The path to the generated video.

        """
        try:

            # Generate low resolution video
            pipe = DiffusionPipeline.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()

            pipe.enable_vae_slicing()
            pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)

            video_frames = pipe(
                self.prompt,
                num_inference_steps=self.num_inference_steps,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames
            ).frames

            # Upscale the video
            pipe = DiffusionPipeline.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            video = [
                Image.fromarray(frame).resize(
                    (self.width, self.height)
                ) for frame in video_frames
            ]

            video_frames = pipe(
                self.prompt,
                video=video,
                strength=self.strength
            ).frames

            # Export the video
            video_path = export_to_video(video_frames, output_video_path=self.output_video_path, output_format=self.output_format)

            print(colored(f'Successfully generated the video at: {video_path}', 'green'))
            return video_path

        except Exception as e:
            print(colored(f'An error occurred: {str(e)}', 'red'))
            logging.error(f'An error occurred: {str(e)}')

