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
    version: str
    allowed_domains: str
    ssl_keyfile_path: str
    ssl_certfile_path: str
    enable_rest: bool = True
    enable_grpc: bool = False
    grpc_port: int = 50051
    transformer_model: str

    @property
    def allowed_cors_origins(self) -> list[str]:
        return (
            self.allowed_domains.strip().split(",")
            if self.allowed_domains.strip()
            else []
        )

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore
