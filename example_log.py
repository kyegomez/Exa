from exa import Logger, Inference


# Initialize the Logger class to monitor GPU usage
devices_to_monitor = [0]  # Replace with the GPU index you want to monitor
logger = Logger(devices=devices_to_monitor)

try:
    logger.start_logging()

    # Perform inference
    for _ in range(10):  # Perform 10 inference iterations as an example
        model = Inference(
            model_id="georgesung/llama2_7b_chat_uncensored",
            quantize=True
        )

        model.run("What is your name")
    input("Press Enter to stop logging and exit...")
except KeyboardInterrupt:
    pass
finally:
    logger.close()