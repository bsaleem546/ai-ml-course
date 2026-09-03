from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_ml"
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "development"
    db_ssl_required: bool = True
    
    groq_api_key: str | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_default_model: str = "openai/gpt-oss-20b"

settings = Settings()