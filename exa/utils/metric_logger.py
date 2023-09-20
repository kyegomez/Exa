import torch
import time
import logging
from termcolor import colored

class Logging:
    """
    Logging class for tracking various metrics during inference.

    Attributes:
        start_time (float): The start time of the inference.
        end_time (float): The end time of the inference.
        num_tokens (int): The number of tokens processed.
        device (str): The device used for inference.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.num_tokens = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.basicConfig(level=logging.DEBUG)

    def start(self):
        """Start the timer and reset the number of tokens."""
        self.start_time = time.time()
        self.num_tokens = 0
        logging.debug('Timer started.')

    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        logging.debug('Timer stopped.')

    def add_tokens(self, num_tokens):
        """Add the number of tokens processed."""
        self.num_tokens += num_tokens
        logging.debug(f'Added {num_tokens} tokens.')

    def get_elapsed_time(self):
        """Get the elapsed time in seconds."""
        return self.end_time - self.start_time

    def get_tokens_per_second(self):
        """Get the number of tokens processed per second."""
        return self.num_tokens / self.get_elapsed_time()

    def get_memory_usage(self):
        """Get the current GPU memory usage in bytes."""
        return torch.cuda.memory_allocated(self.device)

    def get_max_memory_usage(self):
        """Get the maximum GPU memory usage in bytes."""
        return torch.cuda.max_memory_allocated(self.device)

    def get_num_cuda_devices(self):
        """Get the number of available CUDA devices."""
        return torch.cuda.device_count()

    def get_device_name(self):
        """Get the name of the current device."""
        return torch.cuda.get_device_name(self.device)

    def get_device_capability(self):
        """Get the CUDA capability of the current device."""
        return torch.cuda.get_device_capability(self.device)

    def print_summary(self):
        """Print a summary of the metrics."""
        print(colored(f"Elapsed time: {self.get_elapsed_time()} seconds", 'green'))
        print(colored(f"Tokens per second: {self.get_tokens_per_second()}", 'green'))
        print(colored(f"Memory usage: {self.get_memory_usage()} bytes", 'yellow'))
        print(colored(f"Max memory usage: {self.get_max_memory_usage()} bytes", 'yellow'))
        print(colored(f"Number of CUDA devices: {self.get_num_cuda_devices()}", 'blue'))
        print(colored(f"Device name: {self.get_device_name()}", 'blue'))
        print(colored(f"Device capability: {self.get_device_capability()}", 'blue'))

        logging.debug('Printed summary.')