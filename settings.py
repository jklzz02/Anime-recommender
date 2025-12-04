from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

class Environment(str, Enum):
    DEV = "dev"
    DEVELOPMENT = "development"
    PROD = "prod"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self in {Environment.PROD, Environment.PRODUCTION}

    @property
    def is_development(self) -> bool:
        return self in {Environment.DEV, Environment.DEVELOPMENT}


class Settings(BaseSettings):
    host: str
    port: int
    environment: Environment

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()