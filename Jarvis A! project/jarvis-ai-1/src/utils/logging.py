import logging

def setup_logging(log_file: str = 'app.log', level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Parameters:
    log_file (str): The name of the log file where logs will be saved.
    level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
    None
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',  # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )
    logging.info("Logging is set up.")