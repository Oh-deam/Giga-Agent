from langchain_gigachat import GigaChat

from src.pipeline import pipeline
from src.config.config import config


def initialize_llm():
    config.update_access_token()
    scope = 'GIGACHAT_API_CORP'
    return GigaChat(
        credentials=config.AUTHORIZATION_KEY,
        scope=scope,
        model="GigaChat-2-Max",
        verify_ssl_certs=False,
        # model=gc_cfg.get("model", "GigaChat-2-Max"),
        # timeout=int(gc_cfg.get("timeout", 60)),
        # verify_ssl_certs=bool(gc_cfg.get("verify_ssl_certs", False)),
    )

def run():
    model = initialize_llm()
    pipeline(model=model, debug=True)


if __name__ == "__main__":
    run()