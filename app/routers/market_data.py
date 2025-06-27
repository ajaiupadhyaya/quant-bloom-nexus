from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List
import yfinance as yf
from datetime import date

router = APIRouter()

# Pydantic model for the response data
class HistoricalData(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

@router.get("/historical/{symbol}", response_model=List[HistoricalData])
def get_historical_data(
    symbol: str,
    start_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: date = Query(..., description="End date in YYYY-MM-DD format")
):
    """
    Retrieves historical daily stock data for a given symbol.
    """
    try:
        # Download historical data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for the given symbol and date range.")

        # Convert the DataFrame to a list of Pydantic models
        result = []
        for index, row in data.iterrows():
            result.append(HistoricalData(
                date=index.date(),
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume']
            ))
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))