import logging
import os

import colorlog

from .config import get_system_config

_system_config = get_system_config()


def setup_loggers(logs_path: str) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if not os.path.isabs(logs_path):
        logs_path = os.path.join(_system_config.log_dir, logs_path)

    os.makedirs(os.path.dirname(logs_path), exist_ok=True)

    logging.root.setLevel(_system_config.log_level)
    file_handler = logging.FileHandler(logs_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
    )
    logging.root.addHandler(file_handler)

    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
    )
    logging.root.addHandler(stream_handler)

    logging.getLogger("compact_json").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
