from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_ml"
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "development"
    db_ssl_required: bool = True

settings = Settings()