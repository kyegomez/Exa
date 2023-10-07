from exa import Logger, Inference
from nvitop import Device

# Initialize the Logger class to monitor GPU usage
devices_to_monitor = [Device(cuda_device_idx) for cuda_device_idx in [0]]
logger = Logger(devices=devices_to_monitor)

try:
    logger.start_logging()

    # Perform inference
    model = Inference(model_id="openlm-research/open_llama_3b", quantize=True)

    model.run("What is your name")
except KeyboardInterrupt:
    pass
finally:
    logger.close()
