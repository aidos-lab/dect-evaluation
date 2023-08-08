import logging

from typing import Callable, Any
import time
import functools

import wandb


class Logger:
    def __init__(self, dev: bool = True):
        self.dev = dev
        self.wandb = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def wandb_init(self, config):
        if self.dev:
            self.wandb = None
        else:
            self.wandb = wandb.init(
                project=config.project,
                name=config.name,
                tags=config.tags,
            )

    def log(self, msg: str | None = None, params: dict[str, str] | None = None) -> None:
        if msg:
            self.logger.info(msg=msg)

        if params and self.wandb:
            self.wandb.log(params)

    def log_config(self, config):
        if self.wandb:
            self.wandb.config.update(config)
        else:
            raise ValueError()


def timing(logger: Logger, dev: bool = True):
    if dev:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.log(f"Calling {func.__name__}")
                ts = time.time()
                value = func(*args, **kwargs)
                te = time.time()
                logger.log(f"Finished {func.__name__}")
                if logger:
                    logger.log("func:%r took: %2.4f sec" % (func.__name__, te - ts))
                return value

            return wrapper

        return decorator
    else:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                value = func(*args, **kwargs)
                return value

            return wrapper

        return decorator
