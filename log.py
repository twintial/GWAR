import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    # \x1B为十进制27在 ASCLL 里代表ESC
    grey = "\x1b[38;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

# logger.debug("debug message")
# logger.info("info message")
# logger.warning("warning message")
# logger.error("error message")
# logger.critical("critical message")
