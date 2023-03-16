
import logging

_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%Y-%m-%d %H:%M:%S'

LOGGER = logging.getLogger('__main__')  # this is the global logger
LOGGER.setLevel(logging.DEBUG)

def add_log_to_file(log_path):
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    LOGGER.addHandler(file_handler)

    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(formatter)
    LOGGER.addHandler(terminal_handler)

    LOGGER.info("**"*80)