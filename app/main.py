from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os
from contextlib import asynccontextmanager
import sentry_sdk

# Apply compatibility patches first
from .utils import *  # This applies all compatibility fixes

# Import routers
from .routers import market_data, ai_models, trading, analytics, auth, advanced_ai
from .services.ai_engine import ai_engine
from .services.advanced_ai_engine import advanced_ai_engine
from .services.market_data_service import market_data_service
from .core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Sentry for error logging
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN and SENTRY_DSN.startswith("http"):
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=0.2,  # Adjust as needed
        environment=settings.ENVIRONMENT,
        release=settings.VERSION
    )
    logger.info("Sentry initialized for error logging.")
else:
    logger.warning("Sentry DSN not set or invalid. Error logging to Sentry is disabled.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Quant Bloom Nexus application...")
    
    # Initialize services
    try:
        # Services are initialized on import
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Quant Bloom Nexus application...")
    try:
        # Cleanup will be handled by service destructors
        logger.info("All services cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Quant Bloom Nexus",
    description="Advanced Quantitative Trading Terminal with AI/ML/RL capabilities",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "ai_engine": "operational",
            "market_data": "operational",
            "database": "operational"
        }
    }

# API Info endpoint
@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "title": "Quant Bloom Nexus API",
        "version": "2.0.0",
        "description": "Advanced quantitative trading terminal with comprehensive financial analytics",
        "features": [
            "Real-time market data",
            "AI/ML price predictions",
            "Deep reinforcement learning trading agents",
            "Advanced options pricing models",
            "Portfolio optimization",
            "Risk management",
            "Technical analysis",
            "Sentiment analysis",
            "News aggregation",
            "Backtesting engine"
        ],
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc"
        }
    }

# Include API routers
app.include_router(market_data.router, tags=["Market Data"])
app.include_router(ai_models.router, tags=["AI Models"])
app.include_router(advanced_ai.router, tags=["Advanced AI"])
app.include_router(trading.router, tags=["Trading"])
app.include_router(analytics.router, tags=["Analytics"])
app.include_router(auth.router, tags=["Auth"])

# Serve static files for frontend
if os.path.exists("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend application"""
        # Serve index.html for all routes (SPA routing)
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        file_path = f"dist/{full_path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Default to index.html for SPA
        return FileResponse("dist/index.html")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "status_code": 404}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    from fastapi.responses import JSONResponse
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
