from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 4
    chunk_size: int = 900
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
