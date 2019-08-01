""" A convenience log function, behaves similar to print() """
from logging import getLogger


def log(*args, level="info", name=None):
    # TODO basic logger formatting
    logger = getLogger(name=name)

    message = " ".join([str(a) for a in args])

    level = level.lower()
    if level == "exception" or level == "error":
        logger.exception(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "info":
        logger.info(message)
    elif level == "debug":
        logger.debug(message)
    else:
        raise TypeError(
            "level must be one of 'debug', 'info', 'warning', 'error', or 'exception' (case insensitive)"
        )


def get_log_func(name, level="info"):
    def _log(*args, level=level):
        log(*args, name=name, level=level)

    return _log
