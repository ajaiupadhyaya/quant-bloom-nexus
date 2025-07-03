import json
from typing import List, Dict, Union
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta

def calculate_portfolio_greeks(positions_json: str) -> Dict[str, float]:
    """
    Calculates the aggregated Delta, Gamma, Vega, and Theta for a portfolio
    of options and stocks.

    Args:
        positions_json (str): A JSON string representing the portfolio positions.
                              Each position should be a dictionary with the following keys:
                              - 'type': 'stock' or 'option'
                              - For 'stock':
                                - 'symbol': str
                                - 'quantity': float
                                - 'price': float
                              - For 'option':
                                - 'symbol': str
                                - 'quantity': float
                                - 'option_type': 'call' or 'put'
                                - 's': float (underlying asset price)
                                - 'k': float (strike price)
                                - 't': float (time to expiration in years)
                                - 'r': float (risk-free interest rate)
                                - 'sigma': float (volatility)

    Returns:
        Dict[str, float]: A dictionary containing the total portfolio Delta, Gamma, Vega, and Theta.
                          Example: {"total_delta": 150.5, "total_gamma": 2.3, ...}
    """
    positions: List[Dict[str, Union[str, float]]] = json.loads(positions_json)

    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0

    for position in positions:
        pos_type = position.get('type')
        quantity = position.get('quantity', 0.0)

        if pos_type == 'stock':
            # For stocks, Delta is simply the quantity of shares
            total_delta += quantity
            # Other Greeks are zero for stocks

        elif pos_type == 'option':
            option_type = position.get('option_type')
            s = position.get('s')
            k = position.get('k')
            t = position.get('t')
            r = position.get('r')
            sigma = position.get('sigma')

            if None in [option_type, s, k, t, r, sigma]:
                print(f"Warning: Incomplete option data for position: {position}. Skipping.")
                continue

            try:
                # Calculate Greeks for the option
                option_delta = delta(option_type, s, k, t, r, sigma) * quantity
                option_gamma = gamma(option_type, s, k, t, r, sigma) * quantity
                option_vega = vega(option_type, s, k, t, r, sigma) * quantity
                option_theta = theta(option_type, s, k, t, r, sigma) * quantity

                total_delta += option_delta
                total_gamma += option_gamma
                total_vega += option_vega
                total_theta += option_theta

            except Exception as e:
                print(f"Error calculating Greeks for option {position.get('symbol')}: {e}. Skipping.")
                continue
        else:
            print(f"Warning: Unknown position type '{pos_type}'. Skipping position: {position}")

    return {
        "total_delta": total_delta,
        "total_gamma": total_gamma,
        "total_vega": total_vega,
        "total_theta": total_theta,
    }

# Example Usage:
if __name__ == "__main__":
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

    # Convert the list of dictionaries to a JSON string
    sample_positions_json = json.dumps(sample_positions)

    portfolio_greeks = calculate_portfolio_greeks(sample_positions_json)
    print("\nPortfolio Greeks:")
    print(json.dumps(portfolio_greeks, indent=4))

    # Example with missing data
    invalid_positions_json = json.dumps([
        {
            "type": "option",
            "symbol": "MSFT",
            "quantity": 1,
            "option_type": "call",
            "s": 300.0,
            "k": 310.0,
            "t": 0.1,
            "r": 0.01,
            # "sigma": 0.30 # Missing sigma
        }
    ])
    print("\nPortfolio Greeks (with missing data):")
    print(json.dumps(calculate_portfolio_greeks(invalid_positions_json), indent=4))

