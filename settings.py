from pydantic_settings import BaseSettings
from decouple import config


class Settings(BaseSettings):
    """
    Class holding all the necessary config and env settings
    for the experiments.yaml.
    """

    WANDB_API_KEY: str = config("WANDB_API_KEY", cast=str)

    class Config:
        case_sensitive = True


settings = Settings()