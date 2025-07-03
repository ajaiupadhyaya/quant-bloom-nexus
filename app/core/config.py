from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Quant Bloom Nexus Trading Terminal"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]
    
    # Database settings
    DATABASE_URL: str = "postgresql://postgres:quantpass123@localhost:5432/quantdb"
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = "demo-token-replace-with-real"
    INFLUXDB_ORG: str = "quant-org"
    INFLUXDB_BUCKET: str = "trading-data"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: str = "demo"
    POLYGON_API_KEY: str = "demo"
    IEX_CLOUD_API_KEY: str = "demo"
    NEWS_API_KEY: str = "demo"
    FMP_API_KEY: str = "demo"
    OPENAI_API_KEY: str = "demo"
    HUGGINGFACE_TOKEN: str = "demo"
    
    # Trading APIs
    ALPACA_API_KEY: str = "demo"
    ALPACA_API_SECRET: str = "demo"
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # Security
    JWT_SECRET: str = "super-secure-jwt-secret-change-this-in-production"
    ENCRYPTION_KEY: str = "your-32-character-encryption-key-here"
    
    # Application settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # AI/ML settings
    AI_MODEL_CACHE_DIR: str = "/app/models"
    LSTM_MODEL_PATH: str = "/app/models/lstm_price_predictor.pkl"
    TRANSFORMER_MODEL_PATH: str = "/app/models/transformer_sentiment.pkl"
    RL_MODEL_PATH: str = "/app/models/rl_trading_agent.pkl"
    
    # Performance settings
    RATE_LIMIT_PER_MINUTE: int = 1000
    CACHE_TTL: int = 300
    MAX_CONCURRENT_REQUESTS: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def get_database_url(self):
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

settings = Settings()
