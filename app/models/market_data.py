
from pydantic import BaseModel

class MarketData(BaseModel):
    symbol: str
    price: float
    volume: int
