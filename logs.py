import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
timestr = time.strftime("%Y%m%d-%H%M%S")
fh = logging.FileHandler(filename=f"./logs/test-{timestr}.logs", mode="a")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


def log_msg(msg: str):
    logger.info(msg)
