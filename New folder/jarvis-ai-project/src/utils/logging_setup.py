import logging

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.

    Parameters:
    log_level (int): The logging level (default is logging.INFO).
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=log_level
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up.")
    return logger