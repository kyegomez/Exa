from exa import Logger, Inference
from nvitop import Device

# Initialize the Logger class to monitor GPU usage
devices_to_monitor = [Device(cuda_device_idx) for cuda_device_idx in [0]]
logger = Logger(devices=devices_to_monitor)

try:
    logger.start_logging()

    # Perform inference
    for _ in range(10):  # Perform 10 inference iterations as an example
        model = Inference(
            model_id="gpt2-small",
            quantize=True
        )

        model.run("What is your name")
    input("Press Enter to stop logging and exit...")
except KeyboardInterrupt:
    pass
finally:
    logger.close()