
class MarketDataService:
    def get_data(self, symbol: str):
        # Business logic to get market data from a database or external API
        return {"symbol": symbol, "price": 100.0}

market_data_service = MarketDataService()
