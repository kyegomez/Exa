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


# Decorator to log method entry and exit
def log_metadata(cls):
    """
    @log_metadata
    class MyClass:
        def say_hello(self):
            print("hello")
        
        def say_goodbye(self):
            print(f"Goodbye!")
    """
            
    class Wrapped(cls):
        def __init__(self, *args, **kwargs):
            self._logger = logging.getLogger(cls.__name__)
            super().__init__(*args, **kwargs)

        def __getattribute__(self, s):
            attr = super().__getattribute__(s)

            if callable(attr):
                def wrapped(*args, **kwargs):
                    self._logger.debug(colored(f'Entering {s} method', 'blue'))
                    result = attr(*args, **kwargs)
                    self._logger.debug(colored(f'Exiting {s} method', 'blue'))
                    return result
                return wrapped
            else:
                return attr
    return Wrapped
