import logging
from termcolor import colored


# Custom logging formatter for color-coded logging
class CustomFormatter(logging.Formatter):
    format_mappings = {
        logging.DEBUG: {"color": "grey", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        logging.INFO: {"color": "green", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        logging.WARNING: {"color": "yellow", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        logging.ERROR: {"color": "red", "format": "%(asctime)s - %(levelname)s - %(message)s"},
    }

    def format(self, record):
        format_dict = self.format_mappings.get(record.levelno)
        log_message = super().format(record)
        return colored(log_message, format_dict["color"])

# Colored logging setup
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)


class ColoredLogger:
    @staticmethod
    def info(msg):
        logger.info(colored(msg, 'green'))

    @staticmethod
    def warning(msg):
        logger.warning(colored(msg, 'yellow'))

    @staticmethod
    def error(msg):
        logger.error(colored(msg, 'red'))

    @staticmethod
    def debug(msg):
        logger.debug(colored(msg, 'blue'))