

import os
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AlpacaConnection:
    """
    A class to handle interactions with the Alpaca trading API.
    """
    def __init__(self):
        """
        Initializes the Alpaca API connection using credentials from environment variables.
        
        Expects:
        - APCA_API_KEY_ID: Your Alpaca API Key ID.
        - APCA_API_SECRET_KEY: Your Alpaca API Secret Key.
        - APCA_API_BASE_URL: The base URL for the Alpaca API (e.g., https://paper-api.alpaca.markets for paper trading).
        """
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret must be set as environment variables.")

        try:
            self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        except Exception as e:
            raise ConnectionError(f"Failed to establish a connection with Alpaca: {e}")

    def get_account_details(self):
        """
        Checks the connection status and retrieves account details.

        Returns:
            A dictionary with account details if successful, None otherwise.
        """
        try:
            account = self.api.get_account()
            print("Successfully connected to Alpaca.")
            return {
                "account_number": account.account_number,
                "status": account.status,
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
            }
        except APIError as e:
            print(f"API Error checking account status: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def submit_market_order(self, symbol: str, qty: float, side: str):
        """
        Submits a market order to the Alpaca API.

        Args:
            symbol (str): The stock symbol to trade (e.g., 'AAPL').
            qty (float): The number of shares to trade.
            side (str): The order side, either 'buy' or 'sell'.

        Returns:
            The order object if the order was successfully submitted, None otherwise.
        """
        if side not in ['buy', 'sell']:
            print(f"Error: Invalid order side '{side}'. Must be 'buy' or 'sell'.")
            return None

        try:
            print(f"Submitting market {side} order for {qty} shares of {symbol}...")
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"Order submitted successfully. Order ID: {order.id}")
            return order
        except APIError as e:
            # Handle specific API errors
            if "insufficient" in str(e).lower():
                print(f"Error: Insufficient funds to place the order. Details: {e}")
            elif "invalid" in str(e).lower():
                print(f"Error: The order was rejected as invalid. Details: {e}")
            else:
                print(f"API Error submitting order: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while submitting the order: {e}")
            return None

# Example of how to use the AlpacaConnection class:
if __name__ == '__main__':
    # Make sure to set your environment variables in a .env file or directly
    # For example:
    # APCA_API_KEY_ID="YOUR_KEY_ID"
    # APCA_API_SECRET_KEY="YOUR_SECRET_KEY"
    # APCA_API_BASE_URL="https://paper-api.alpaca.markets"

    try:
        alpaca = AlpacaConnection()
        
        # 1. Check connection and get account details
        account_info = alpaca.get_account_details()
        if account_info:
            print("\nAccount Information:")
            for key, value in account_info.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")

        # 2. Submit a market order (use with caution, even in paper trading)
        # Example: Buy 1 share of AAPL
        # order_details = alpaca.submit_market_order('AAPL', 1, 'buy')
        # if order_details:
        #     print("\nOrder Details:")
        #     print(f"  ID: {order_details.id}")
        #     print(f"  Symbol: {order_details.symbol}")
        #     print(f"  Qty: {order_details.qty}")
        #     print(f"  Side: {order_details.side}")
        #     print(f"  Status: {order_details.status}")

        # Example of an order that might fail (e.g., insufficient funds)
        # print("\n--- Testing Error Handling ---")
        # alpaca.submit_market_order('AAPL', 1000000, 'buy') # Likely to fail

    except (ValueError, ConnectionError) as e:
        print(f"Failed to initialize Alpaca connection: {e}")

