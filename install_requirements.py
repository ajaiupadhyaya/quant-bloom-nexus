#!/usr/bin/env python3
"""
Safe installation script for institutional trading terminal requirements.
This script handles platform-specific issues and provides fallbacks for problematic packages.
"""

import subprocess
import platform


def run_command(command, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def install_package(package, alternative=None):
    """Install a package with fallback to alternative if it fails."""
    print(f"Installing {package}...")
    success, stdout, stderr = run_command(f"pip install {package}", check=False)

    if not success:
        print(f"  ❌ Failed to install {package}")
        if stderr:
            print(f"  Error: {stderr.strip()}")

        if alternative:
            print(f"  🔄 Trying alternative: {alternative}")
            success, stdout, stderr = run_command(
                f"pip install {alternative}", check=False
            )
            if success:
                print(f"  ✅ Successfully installed {alternative}")
                return True
            else:
                print(f"  ❌ Alternative also failed: {alternative}")
                return False
        return False
    else:
        print(f"  ✅ Successfully installed {package}")
        return True


def main():
    print("🚀 Installing Institutional Trading Terminal Requirements")
    print("=" * 60)

    # Detect platform
    system = platform.system()
    print(f"Detected platform: {system}")

    # Core packages that should install on most systems
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "python-dateutil>=2.8.0",
        "pytz>=2023.3",
        "yfinance>=0.2.18",
        "pandas-datareader>=0.10.0",
        "backtrader>=1.9.78",
        "empyrical>=0.5.5",
        "statsmodels>=0.14.0",
        "plotly>=5.15.0",
        "dash>=2.14.0",
        "streamlit>=1.25.0",
        "redis>=4.5.0",
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=2.0.0",
        "pymongo>=4.3.0",
        "websocket-client>=1.6.0",
        "beautifulsoup4>=4.12.0",
        "nltk>=3.8.0",
        "textblob>=0.17.0",
        "jupyter>=1.0.0",
        "ipython>=8.10.0",
        "pytest>=7.2.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "rich>=13.3.0",
        "psutil>=5.9.0",
    ]

    # ML/AI packages with alternatives
    ml_packages = [
        ("tensorflow>=2.13.0", "tensorflow-cpu>=2.13.0"),
        ("torch>=2.0.0", "torch>=1.13.0"),
        ("torchvision>=0.15.0", "torchvision>=0.14.0"),
        ("transformers>=4.30.0", "transformers>=4.25.0"),
        ("scikit-learn>=1.3.0", "scikit-learn>=1.2.0"),
    ]

    # Financial packages
    financial_packages = [
        "alpha-vantage>=2.3.0",
        "ccxt>=4.0.0",
        "quandl>=3.7.0",
        "zipline-reloaded>=3.0.0",
        "pyfolio>=0.9.2",
        "pandas-ta>=0.3.14",
        "stockstats>=0.6.0",
        "finta>=1.3",
        "riskfolio-lib>=5.0.0",
        "pypfopt>=1.5.0",
        "vectorbt>=0.25.0",
    ]

    # Specialized packages that might need system dependencies
    specialized_packages = [
        "influxdb-client>=1.35.0",
        "kafka-python>=2.0.0",
        "celery>=5.3.0",
        "spacy>=3.6.0",
        "networkx>=3.0.0",
        "python-igraph>=0.10.0",
        "cryptography>=40.0.0",
        "bcrypt>=4.0.0",
        "pyjwt>=2.8.0",
        "boto3>=1.26.0",
        "httpx>=0.24.0",
    ]

    failed_packages = []

    # Install core packages
    print("\n📦 Installing core packages...")
    for package in core_packages:
        if not install_package(package):
            failed_packages.append(package)

    # Install ML packages with alternatives
    print("\n🤖 Installing ML/AI packages...")
    for package, alternative in ml_packages:
        if not install_package(package, alternative):
            failed_packages.append(package)

    # Install financial packages
    print("\n💰 Installing financial packages...")
    for package in financial_packages:
        if not install_package(package):
            failed_packages.append(package)

    # Install specialized packages
    print("\n🔧 Installing specialized packages...")
    for package in specialized_packages:
        if not install_package(package):
            failed_packages.append(package)

    # Platform-specific packages
    if system != "Darwin":  # Not macOS
        print("\n🖥️ Installing platform-specific packages...")
        platform_packages = [
            "cupy-cuda12x>=12.0.0",  # CUDA support
        ]
        for package in platform_packages:
            if not install_package(package):
                failed_packages.append(package)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Installation Summary")
    print("=" * 60)

    if failed_packages:
        print(f"❌ {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\n💡 These packages can be installed manually later:")
        print("   pip install <package_name>")
    else:
        print("✅ All packages installed successfully!")

    print("\n🎯 Next steps:")
    print("1. Test the installation with: python -c 'import pandas, numpy, sklearn'")
    print("2. For TA-Lib support, install system dependencies:")
    if system == "Darwin":
        print("   brew install ta-lib")
    elif system == "Linux":
        print("   sudo apt-get install libta-lib-dev")
    print("   Then: pip install TA-Lib")
    print("3. For QuantLib support:")
    if system == "Darwin":
        print("   brew install quantlib")
    elif system == "Linux":
        print("   sudo apt-get install libquantlib0-dev")
    print("   Then: pip install QuantLib")


if __name__ == "__main__":
    main()
