#!/usr/bin/env python3
"""
Test script to verify the installation of core packages for the trading terminal.
"""


def test_core_packages():
    """Test core data science packages."""
    print("Testing core data science packages...")

    try:
        import numpy as np

        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy:", str(e))

    try:
        import pandas as pd

        print("✅ Pandas:", pd.__version__)
    except ImportError as e:
        print("❌ Pandas:", str(e))

    try:
        import scipy

        print("✅ SciPy:", scipy.__version__)
    except ImportError as e:
        print("❌ SciPy:", str(e))

    try:
        import sklearn

        print("✅ Scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("❌ Scikit-learn:", str(e))

    try:
        import matplotlib

        print("✅ Matplotlib:", matplotlib.__version__)
    except ImportError as e:
        print("❌ Matplotlib:", str(e))


def test_financial_packages():
    """Test financial analysis packages."""
    print("\nTesting financial packages...")

    try:
        import yfinance as yf

        print("✅ yfinance:", yf.__version__)
    except ImportError as e:
        print("❌ yfinance:", str(e))

    try:
        import backtrader as bt

        print("✅ Backtrader:", bt.__version__)
    except ImportError as e:
        print("❌ Backtrader:", str(e))

    try:
        import pandas_ta as ta

        print("✅ pandas_ta:", ta.__version__)
    except ImportError as e:
        print("❌ pandas_ta:", str(e))


def test_web_packages():
    """Test web and API packages."""
    print("\nTesting web packages...")

    try:
        import fastapi

        print("✅ FastAPI:", fastapi.__version__)
    except ImportError as e:
        print("❌ FastAPI:", str(e))

    try:
        import requests

        print("✅ Requests:", requests.__version__)
    except ImportError as e:
        print("❌ Requests:", str(e))

    try:
        import plotly

        print("✅ Plotly:", plotly.__version__)
    except ImportError as e:
        print("❌ Plotly:", str(e))


def test_data_functionality():
    """Test basic data functionality."""
    print("\nTesting basic functionality...")

    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf

        # Test data creation
        df = pd.DataFrame(
            {
                "price": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )
        print("✅ Data creation works")

        # Test simple calculation
        df["sma"] = df["price"].rolling(10).mean()
        print("✅ Technical indicators work")

        # Test yfinance (if internet available)
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            print("✅ Market data access works")
        except Exception:
            print("⚠️ Market data access failed (might be network issue)")

    except Exception as e:
        print("❌ Basic functionality test failed:", str(e))


def main():
    print("🚀 Testing Trading Terminal Package Installation")
    print("=" * 60)

    test_core_packages()
    test_financial_packages()
    test_web_packages()
    test_data_functionality()

    print("\n" + "=" * 60)
    print("✅ Installation test complete!")
    print(
        "\n💡 If you see ❌ errors above, those packages need to be installed manually."
    )
    print("   Example: pip install <package_name>")


if __name__ == "__main__":
    main()
