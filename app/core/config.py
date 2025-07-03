from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Quant Bloom Nexus Trading Terminal"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS settings - use string that can be split
    CORS_ORIGINS_STR: str = "http://localhost:3000,http://localhost:5173,http://localhost:8080"
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS_STR.split(",")]
    
    # Database settings
    DATABASE_URL: str = "postgresql://postgres:quantpass123@localhost:5432/quantdb"
    POSTGRES_USER: str = "quantuser"
    POSTGRES_PASSWORD: str = "quantpass"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "quantdb"
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = "demo-token-replace-with-real"
    INFLUXDB_ORG: str = "quant-org"
    INFLUXDB_BUCKET: str = "trading-data"
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: str = "demo"
    POLYGON_API_KEY: str = "demo"
    IEX_CLOUD_API_KEY: str = "demo"
    NEWS_API_KEY: str = "demo"
    FMP_API_KEY: str = "demo"
    OPENAI_API_KEY: str = "demo"
    HUGGINGFACE_TOKEN: str = "demo"
    HUGGINGFACE_API_KEY: str = "your-huggingface-api-key-here"
    ANTHROPIC_API_KEY: str = "your-anthropic-api-key-here"
    TIINGO_API_KEY: str = "47131c77a1468f9d55d992a0316a197aad78d01c"
    QUANDL_API_KEY: str = "xoh1Yn7XxQYbUkLczo-H"
    TWELVE_DATA_API_KEY: str = "2a45279f266f448caf2f47125d19a183"
    FINNHUB_API_KEY: str = "d14dflhr01qrqeatld5gd14dflhr01qrqeatld60"
    FRED_API_KEY: str = "ac30a4d95a47cc0cba09469cb4e7be6a"
    BENZINGA_API_KEY: str = "bz.REQXDUDU6RSKXRB4RSZMG525BE6SCPIO"
    
    # Trading APIs
    ALPACA_API_KEY: str = "demo"
    ALPACA_API_SECRET: str = "demo"
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    BROKER_API_KEY: str = "your-broker-api-key-here"
    BROKER_SECRET_KEY: str = "your-broker-secret-key-here"
    TRADE_EXECUTION_MODE: str = "paper"
    MAX_TRADE_RISK: str = "0.02"
    
    # Security
    JWT_SECRET: str = "super-secure-jwt-secret-change-this-in-production"
    ENCRYPTION_KEY: str = "your-32-character-encryption-key-here"
    
    # Application settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    FRONTEND_URL: str = "http://localhost:3000"
    API_GATEWAY: str = "http://localhost:8000/api"
    
    # AI/ML settings
    AI_MODEL_CACHE_DIR: str = "/app/models"
    LSTM_MODEL_PATH: str = "/app/models/lstm_price_predictor.pkl"
    TRANSFORMER_MODEL_PATH: str = "/app/models/transformer_sentiment.pkl"
    RL_MODEL_PATH: str = "/app/models/rl_trading_agent.pkl"
    TRANSFORMERS_CACHE: str = "/tmp/transformers_cache"
    YFINANCE_API_ENABLED: str = "true"
    
    # Performance settings
    RATE_LIMIT_PER_MINUTE: int = 1000
    CACHE_TTL: int = 300
    MAX_CONCURRENT_REQUESTS: int = 100
    
    # Monitoring and notifications
    SENTRY_DSN: str = "your-sentry-dsn-here"
    SLACK_WEBHOOK_URL: str = "your-slack-webhook-url"
    DISCORD_WEBHOOK_URL: str = "your-discord-webhook-url"
    
    # WebSocket settings
    WEBSOCKET_PROVIDER: str = "wss://example-feed-provider.com"
    WEBSOCKET_API_KEY: str = "your-websocket-provider-api-key"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment

    def get_database_url(self):
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
