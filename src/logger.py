import logging
import sys

class ColorFormatter(logging.Formatter):
    blue = "\x1b[34m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMAT_TEMPLATE = "%(asctime)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: FORMAT_TEMPLATE,
        logging.INFO: FORMAT_TEMPLATE,
        logging.WARNING: yellow + FORMAT_TEMPLATE + reset,
        logging.ERROR: red + FORMAT_TEMPLATE + reset,
        logging.CRITICAL: bold_red + FORMAT_TEMPLATE + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT_TEMPLATE)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def get_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        
    return logger
