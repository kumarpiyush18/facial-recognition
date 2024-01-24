import logging
import sys


def get_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    format_style = "%(asctime)s\t|\t%(name)s:%(lineno)d\t|\t%(levelname)s\t|\t%(filename)s\t|\tfunc:%(funcName)s\t|\t%(message)s"

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_style))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
