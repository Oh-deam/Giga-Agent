import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from langchain_gigachat import GigaChat

from src.pipeline import pipeline
from src.config.config import config


def disable_proxy_env():
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    for key in proxy_keys:
        os.environ.pop(key, None)


def initialize_llm():
    return GigaChat(
        credentials=config.AUTHORIZATION_KEY,
        scope=config.GIGACHAT_SCOPE,
        model=config.GIGACHAT_MODEL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        timeout=config.GIGACHAT_TIMEOUT,
        max_retries=config.GIGACHAT_MAX_RETRIES,
        retry_backoff_factor=config.GIGACHAT_RETRY_BACKOFF_FACTOR,
    )


def configure_logging():
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = log_dir / f"run_{run_stamp}.log"
    latest_log_path = log_dir / "latest.log"
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"

    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL, format=log_format, backtrace=False, diagnose=False)
    logger.add(run_log_path, level=config.LOG_LEVEL, format=log_format, backtrace=False, diagnose=False, encoding="utf-8")
    logger.add(latest_log_path, level=config.LOG_LEVEL, format=log_format, backtrace=False, diagnose=False, encoding="utf-8", mode="w")
    logger.info(f"Log files: run={run_log_path} latest={latest_log_path}")


def run():
    disable_proxy_env()
    configure_logging()
    model = initialize_llm()
    pipeline(model=model, debug=True)


if __name__ == "__main__":
    run()
