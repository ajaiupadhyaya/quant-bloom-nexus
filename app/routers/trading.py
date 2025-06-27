
from fastapi import APIRouter, HTTPException
import json
from app.services.greeks_calculator import calculate_portfolio_greeks

router = APIRouter()

@router.post("/trading/order")
def create_order():
    # Your logic to create a trading order
    return {"status": "order created"}

@router.get("/portfolio/greeks")
def get_portfolio_greeks():
    """
    Calculates and returns the aggregated Delta, Gamma, Vega, and Theta for a sample portfolio.
    """
    sample_positions = [
        {
            "type": "stock",
            "symbol": "AAPL",
            "quantity": 100,
            "price": 170.0
        },
        {
            "type": "option",
            "symbol": "AAPL",
            "quantity": 2,
            "option_type": "call",
            "s": 170.0,  # Underlying price
            "k": 175.0,  # Strike price
            "t": 0.25,   # Time to expiration (0.25 years = 3 months)
            "r": 0.01,   # Risk-free rate
            "sigma": 0.20 # Volatility
        },
        {
            "type": "option",
            "symbol": "GOOGL",
            "quantity": -1, # Short put option
            "option_type": "put",
            "s": 120.0,
            "k": 115.0,
            "t": 0.5,
            "r": 0.01,
            "sigma": 0.25
        }
    ]

    try:
        portfolio_greeks = calculate_portfolio_greeks(json.dumps(sample_positions))
        return portfolio_greeks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio Greeks: {e}")
