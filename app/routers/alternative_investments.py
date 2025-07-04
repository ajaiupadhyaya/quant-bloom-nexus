from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..services.alternative_investments_service import alternative_investments_service, InvestmentType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alternative-investments", tags=["Alternative Investments"])

# =================== REQUEST MODELS ===================

class InvestmentAnalysisRequest(BaseModel):
    investment_type: str
    identifier: str
    additional_params: Optional[Dict[str, Any]] = {}

class HedgeFundAnalysisRequest(BaseModel):
    fund_identifier: str
    include_style_analysis: bool = True
    include_risk_analysis: bool = True
    include_peer_comparison: bool = True

class PrivateEquityAnalysisRequest(BaseModel):
    fund_identifier: str
    include_pme_analysis: bool = True
    include_cash_flow_analysis: bool = True
    vintage_year: Optional[int] = None

class RealEstateAnalysisRequest(BaseModel):
    property_identifier: str
    include_market_analysis: bool = True
    include_comparable_analysis: bool = True
    property_type: Optional[str] = None

class CommodityAnalysisRequest(BaseModel):
    commodity_symbol: str
    include_futures_analysis: bool = True
    include_fundamentals: bool = True
    include_seasonal_analysis: bool = True

class CryptocurrencyAnalysisRequest(BaseModel):
    crypto_symbol: str
    include_onchain_analysis: bool = True
    include_defi_analysis: bool = True
    include_network_analysis: bool = True

class ESGAnalysisRequest(BaseModel):
    symbol: str
    include_peer_comparison: bool = True
    include_controversy_analysis: bool = True
    include_climate_analysis: bool = True

# =================== HEDGE FUND ANALYTICS ===================

@router.post("/hedge-funds/analyze")
async def analyze_hedge_fund(request: HedgeFundAnalysisRequest):
    """Comprehensive hedge fund analysis"""
    try:
        logger.info(f"Analyzing hedge fund: {request.fund_identifier}")
        
        result = await alternative_investments_service.get_hedge_fund_analytics(
            request.fund_identifier
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "fund_identifier": request.fund_identifier,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Hedge fund analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/hedge-funds/{fund_identifier}")
async def get_hedge_fund_analytics(fund_identifier: str):
    """Get hedge fund analytics by identifier"""
    try:
        result = await alternative_investments_service.get_hedge_fund_analytics(fund_identifier)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get hedge fund analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# =================== PRIVATE EQUITY ANALYTICS ===================

@router.post("/private-equity/analyze")
async def analyze_private_equity(request: PrivateEquityAnalysisRequest):
    """Comprehensive private equity analysis"""
    try:
        logger.info(f"Analyzing private equity fund: {request.fund_identifier}")
        
        result = await alternative_investments_service.get_private_equity_analytics(
            request.fund_identifier
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "fund_identifier": request.fund_identifier,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Private equity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/private-equity/{fund_identifier}")
async def get_private_equity_analytics(fund_identifier: str):
    """Get private equity analytics by identifier"""
    try:
        result = await alternative_investments_service.get_private_equity_analytics(fund_identifier)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get private equity analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# =================== REAL ESTATE ANALYTICS ===================

@router.post("/real-estate/analyze")
async def analyze_real_estate(request: RealEstateAnalysisRequest):
    """Comprehensive real estate analysis"""
    try:
        logger.info(f"Analyzing real estate property: {request.property_identifier}")
        
        result = await alternative_investments_service.get_real_estate_analytics(
            request.property_identifier
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "property_identifier": request.property_identifier,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Real estate analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/real-estate/{property_identifier}")
async def get_real_estate_analytics(property_identifier: str):
    """Get real estate analytics by identifier"""
    try:
        result = await alternative_investments_service.get_real_estate_analytics(property_identifier)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get real estate analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# =================== COMMODITY ANALYTICS ===================

@router.post("/commodities/analyze")
async def analyze_commodity(request: CommodityAnalysisRequest):
    """Comprehensive commodity analysis"""
    try:
        logger.info(f"Analyzing commodity: {request.commodity_symbol}")
        
        result = await alternative_investments_service.get_commodity_analytics(
            request.commodity_symbol
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "commodity_symbol": request.commodity_symbol,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Commodity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/commodities/{commodity_symbol}")
async def get_commodity_analytics(commodity_symbol: str):
    """Get commodity analytics by symbol"""
    try:
        result = await alternative_investments_service.get_commodity_analytics(commodity_symbol)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get commodity analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# =================== CRYPTOCURRENCY ANALYTICS ===================

@router.post("/crypto/analyze")
async def analyze_cryptocurrency(request: CryptocurrencyAnalysisRequest):
    """Comprehensive cryptocurrency analysis"""
    try:
        logger.info(f"Analyzing cryptocurrency: {request.crypto_symbol}")
        
        result = await alternative_investments_service.get_cryptocurrency_analytics(
            request.crypto_symbol
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "crypto_symbol": request.crypto_symbol,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cryptocurrency analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/crypto/{crypto_symbol}")
async def get_cryptocurrency_analytics(crypto_symbol: str):
    """Get cryptocurrency analytics by symbol"""
    try:
        result = await alternative_investments_service.get_cryptocurrency_analytics(crypto_symbol)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cryptocurrency analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# =================== ESG ANALYTICS ===================

@router.post("/esg/analyze")
async def analyze_esg(request: ESGAnalysisRequest):
    """Comprehensive ESG analysis"""
    try:
        logger.info(f"Analyzing ESG metrics for: {request.symbol}")
        
        result = await alternative_investments_service.get_esg_analytics(
            request.symbol
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ESG analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/esg/{symbol}")
async def get_esg_analytics(symbol: str):
    """Get ESG analytics by symbol"""
    try:
        result = await alternative_investments_service.get_esg_analytics(symbol)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ESG analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# =================== COMPREHENSIVE ANALYSIS ===================

@router.post("/analyze")
async def comprehensive_analysis(request: InvestmentAnalysisRequest):
    """Comprehensive analysis for any alternative investment type"""
    try:
        logger.info(f"Comprehensive analysis for {request.investment_type}: {request.identifier}")
        
        # Convert string to enum
        try:
            investment_type = InvestmentType(request.investment_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid investment type: {request.investment_type}. "
                       f"Valid types: {[t.value for t in InvestmentType]}"
            )
        
        result = await alternative_investments_service.get_comprehensive_alternative_investments_analysis(
            investment_type, request.identifier
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "investment_type": request.investment_type,
            "identifier": request.identifier,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# =================== UTILITY ENDPOINTS ===================

@router.get("/investment-types")
async def get_investment_types():
    """Get all supported investment types"""
    return {
        "investment_types": [
            {
                "type": t.value,
                "description": f"{t.value.replace('_', ' ').title()} Analytics"
            }
            for t in InvestmentType
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Alternative Investments Analytics",
        "timestamp": datetime.now().isoformat(),
        "supported_types": [t.value for t in InvestmentType]
    } 