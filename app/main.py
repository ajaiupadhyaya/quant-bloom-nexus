
from fastapi import FastAPI
from app.core.config import settings
from app.db.session import engine
from app.db.base import Base
from app.routers import market_data, trading, ai_models

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(market_data.router, prefix=settings.API_V1_STR, tags=["market_data"])
app.include_router(trading.router, prefix=settings.API_V1_STR, tags=["trading"])
app.include_router(ai_models.router, prefix=settings.API_V1_STR, tags=["ai_models"])
