import logging
from termcolor import colored

class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter for color-coded logging.

    Provides a custom format for logs based on the level of logging.
    Each logging level has its own associated color to easily distinguish
    between different log messages.

    Attributes:
        format_mappings (dict): Mapping of logging levels to their associated color and format.
    
    ###########
    import logging
    from exa import CustomFormatter

    logging.basicConfig(level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logger = logging.getLogger("CustomFormatterExample")
    logger.addHandler(handler)

    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")
    """

    format_mappings = {
        logging.DEBUG: {"color": "grey", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        logging.INFO: {"color": "green", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        logging.WARNING: {"color": "yellow", "format": "%(asctime)s - %(levelname)s - %(message)s"},
        logging.ERROR: {"color": "red", "format": "%(asctime)s - %(levelname)s - %(message)s"},
    }

    def format(self, record):
        """
        Format the log message.

        Args:
            record (LogRecord): The record to be formatted.

        Returns:
            str: Formatted log message with the appropriate color.
        """
        format_dict = self.format_mappings.get(record.levelno)
        log_message = super().format(record)
        return colored(log_message, format_dict["color"])
    
    

class ColoredLogger:
    """
    Utility class for colored logging.

    Provides static methods to log messages with colors associated with
    their log levels. The actual coloring is handled by the CustomFormatter.

    from exa import ColoredLogger

    ColoredLogger.info("This is an info message from ColoredLogger.")
    ColoredLogger.warning("This is a warning message from ColoredLogger.")
    ColoredLogger.error("This is an error message from ColoredLogger.")
    ColoredLogger.debug("This is a debug message from ColoredLogger.")
    """

    @staticmethod
    def info(msg):
        """
        Log an info level message.

        Args:
            msg (str): The message to be logged.
        """
        logger.info(msg)

    @staticmethod
    def warning(msg):
        """
        Log a warning level message.

        Args:
            msg (str): The message to be logged.
        """
        logger.warning(msg)

    @staticmethod
    def error(msg):
        """
        Log an error level message.

        Args:
            msg (str): The message to be logged.
        """
        logger.error(msg)

    @staticmethod
    def debug(msg):
        """
        Log a debug level message.

        Args:
            msg (str): The message to be logged.
        """
        logger.debug(msg)

def log_method_calls(cls):
    """
    Decorator to log method entry and exit for classes.
    
    When a method of a decorated class is called, this decorator logs the 
    method's name upon entering and exiting the method. Useful for debugging
    and tracking the flow of the program.
    
    ```
    from exa import log_metadata

    @log_metadata
    class MyClass:
        def say_hello(self):
            print("Hello from MyClass!")

        def say_goodbye(self):
            print("Goodbye from MyClass!")

    # Using MyClass with log_metadata decorator
    sample_instance = MyClass()
    sample_instance.say_hello()
    sample_instance.say_goodbye()
    ```
    
    Args:
        cls (type): The class to be wrapped.

    Returns:
        type: Wrapped class with added logging functionality.
    """

    class Wrapped(cls):
        """Wrapper class to intercept method calls and log them."""

        def __init__(self, *args, **kwargs):
            self._logger = logging.getLogger(cls.__name__)
            super().__init__(*args, **kwargs)

        def __getattribute__(self, s):
            """
            Intercept method calls to log them.

            Args:
                s (str): The name of the attribute or method to be accessed.

            Returns:
                Any: The wrapped method with added logging or the original attribute.
            """
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

# Set up the logger
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
