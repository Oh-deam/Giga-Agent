from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    AUTHORIZATION_KEY: str = Field(
        validation_alias=AliasChoices("GIGACHAT_CREDENTIALS", "AUTHORIZATION_KEY")
    )
    GIGACHAT_SCOPE: str = Field(
        default="GIGACHAT_API_CORP",
        validation_alias=AliasChoices("GIGACHAT_SCOPE"),
    )
    GIGACHAT_MODEL: str = "GigaChat-2-Max"
    GIGACHAT_VERIFY_SSL_CERTS: bool = False
    GIGACHAT_TIMEOUT: float = 180.0
    GIGACHAT_MAX_RETRIES: int = 4
    GIGACHAT_RETRY_BACKOFF_FACTOR: float = 2.0
    FEATURE_SEARCH_ROUNDS: int = 1 # он всеравно какбы второй раз запускает кэтбуст, надо либо убрать либо придумать чето
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


config = Config()
