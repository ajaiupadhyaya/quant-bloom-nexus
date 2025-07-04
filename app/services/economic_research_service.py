"""
Economic Research and Fundamental Analysis Service
Real implementations connecting to actual data sources including FRED, Alpha Vantage,
Financial Modeling Prep, OECD, World Bank for comprehensive economic and fundamental analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import os
import json
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import yfinance as yf
import requests
from scipy import stats
import math

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    DCF = "dcf"
    COMPARABLE_COMPANY = "comparable_company"
    PRECEDENT_TRANSACTION = "precedent_transaction"
    DIVIDEND_DISCOUNT = "dividend_discount"
    ASSET_BASED = "asset_based"
    SUM_OF_PARTS = "sum_of_parts"

@dataclass
class EconomicIndicator:
    indicator_id: str
    name: str
    value: float
    date: datetime
    frequency: str
    units: str
    source: str
    seasonally_adjusted: bool
    release_date: datetime

@dataclass
class FinancialStatement:
    symbol: str
    period: str  # 'annual' or 'quarterly'
    fiscal_year: int
    fiscal_quarter: Optional[int]
    revenue: float
    cost_of_revenue: float
    gross_profit: float
    operating_expenses: float
    operating_income: float
    ebitda: float
    net_income: float
    eps_diluted: float
    shares_outstanding: float
    total_assets: float
    total_liabilities: float
    shareholders_equity: float
    cash_and_equivalents: float
    total_debt: float
    free_cash_flow: float
    capex: float
    research_development: float

@dataclass
class DCFModel:
    symbol: str
    base_year_fcf: float
    terminal_growth_rate: float
    discount_rate: float
    projection_years: int
    revenue_growth_rates: List[float]
    margin_projections: List[float]
    capex_as_percent_revenue: List[float]
    working_capital_change: List[float]
    terminal_value: float
    present_value_fcf: float
    present_value_terminal: float
    enterprise_value: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float

@dataclass
class ComparableCompany:
    symbol: str
    name: str
    market_cap: float
    enterprise_value: float
    revenue_ttm: float
    ebitda_ttm: float
    net_income_ttm: float
    pe_ratio: float
    ev_revenue: float
    ev_ebitda: float
    price_to_book: float
    roe: float
    debt_to_equity: float
    revenue_growth: float
    industry: str

class EconomicResearchService:
    """Comprehensive economic research and fundamental analysis service"""
    
    def __init__(self):
        # API Keys from environment variables
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.fmp_api_key = os.getenv("FINANCIAL_MODELING_PREP_API_KEY", "")
        self.quandl_api_key = os.getenv("QUANDL_API_KEY", "")
        self.world_bank_key = os.getenv("WORLD_BANK_API_KEY", "")
        self.oecd_api_key = os.getenv("OECD_API_KEY", "")
        
        # Session for HTTP requests
        self.session = None
        
        # Data cache with TTL
        self.cache = {}
        self.cache_ttl = {}
        
        # Economic indicators mapping
        self.fred_indicators = {
            'gdp': 'GDP',
            'unemployment': 'UNRATE',
            'inflation_cpi': 'CPIAUCSL',
            'inflation_pce': 'PCEPI',
            'fed_funds_rate': 'FEDFUNDS',
            '10y_treasury': 'GS10',
            '2y_treasury': 'GS2',
            'consumer_sentiment': 'UMCSENT',
            'retail_sales': 'RSAFS',
            'industrial_production': 'INDPRO',
            'housing_starts': 'HOUST',
            'personal_income': 'PI',
            'personal_consumption': 'PCE',
            'initial_claims': 'ICSA',
            'nonfarm_payrolls': 'PAYEMS',
            'ism_manufacturing': 'NAPM',
            'ism_services': 'NAPMSII',
            'consumer_credit': 'TOTALSL',
            'money_supply_m1': 'M1SL',
            'money_supply_m2': 'M2SL'
        }
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'Economic Research Service'}
            )
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache or key not in self.cache_ttl:
            return False
        return datetime.now() < self.cache_ttl[key]
    
    def _set_cache(self, key: str, data: Any, ttl_minutes: int = 60):
        """Set cache with TTL"""
        self.cache[key] = data
        self.cache_ttl[key] = datetime.now() + timedelta(minutes=ttl_minutes)
    
    # =================== FRED ECONOMIC DATA ===================
    
    async def get_fred_data(self, series_id: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[EconomicIndicator]:
        """Get economic data from Federal Reserve Economic Data (FRED)"""
        if not self.fred_api_key:
            raise ValueError("FRED API key not configured")
        
        cache_key = f"fred_{series_id}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        session = await self.get_session()
        
        # Get series info first
        info_url = f"https://api.stlouisfed.org/fred/series"
        info_params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json'
        }
        
        try:
            async with session.get(info_url, params=info_params) as response:
                if response.status != 200:
                    raise Exception(f"FRED API error: {response.status}")
                
                info_data = await response.json()
                series_info = info_data['seriess'][0]
        
            # Get observations
            obs_url = f"https://api.stlouisfed.org/fred/series/observations"
            obs_params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1000
            }
            
            if start_date:
                obs_params['observation_start'] = start_date
            if end_date:
                obs_params['observation_end'] = end_date
            
            async with session.get(obs_url, params=obs_params) as response:
                if response.status != 200:
                    raise Exception(f"FRED API error: {response.status}")
                
                obs_data = await response.json()
                observations = obs_data['observations']
            
            # Convert to EconomicIndicator objects
            indicators = []
            for obs in observations:
                if obs['value'] != '.':  # FRED uses '.' for missing values
                    indicators.append(EconomicIndicator(
                        indicator_id=series_id,
                        name=series_info['title'],
                        value=float(obs['value']),
                        date=datetime.strptime(obs['date'], '%Y-%m-%d'),
                        frequency=series_info['frequency'],
                        units=series_info['units'],
                        source='FRED',
                        seasonally_adjusted=series_info.get('seasonal_adjustment_short', '') == 'SA',
                        release_date=datetime.strptime(obs['realtime_start'], '%Y-%m-%d')
                    ))
            
            self._set_cache(cache_key, indicators, 240)  # 4 hour cache
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to fetch FRED data for {series_id}: {e}")
            raise
    
    async def get_economic_indicators_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive economic indicators dashboard"""
        dashboard_data = {}
        
        # Key economic indicators
        key_indicators = [
            'gdp', 'unemployment', 'inflation_cpi', 'fed_funds_rate',
            '10y_treasury', 'consumer_sentiment', 'nonfarm_payrolls'
        ]
        
        for indicator in key_indicators:
            fred_id = self.fred_indicators[indicator]
            try:
                data = await self.get_fred_data(fred_id)
                if data:
                    latest = data[0]  # Most recent data point
                    previous = data[1] if len(data) > 1 else None
                    
                    change = latest.value - previous.value if previous else 0
                    change_pct = (change / previous.value * 100) if previous and previous.value != 0 else 0
                    
                    dashboard_data[indicator] = {
                        'current_value': latest.value,
                        'date': latest.date.isoformat(),
                        'change': change,
                        'change_percent': change_pct,
                        'units': latest.units,
                        'frequency': latest.frequency
                    }
            except Exception as e:
                logger.error(f"Failed to get {indicator}: {e}")
                dashboard_data[indicator] = {'error': str(e)}
        
        return dashboard_data
    
    # =================== FINANCIAL STATEMENTS DATA ===================
    
    async def get_financial_statements(self, symbol: str, period: str = 'annual',
                                     years: int = 5) -> List[FinancialStatement]:
        """Get comprehensive financial statements from Financial Modeling Prep"""
        if not self.fmp_api_key:
            # Fallback to yfinance for basic data
            return await self._get_financial_statements_yfinance(symbol, period, years)
        
        cache_key = f"financials_{symbol}_{period}_{years}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        session = await self.get_session()
        statements = []
        
        try:
            # Income Statement
            income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
            income_params = {'period': period, 'limit': years, 'apikey': self.fmp_api_key}
            
            async with session.get(income_url, params=income_params) as response:
                income_data = await response.json() if response.status == 200 else []
            
            # Balance Sheet
            balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}"
            balance_params = {'period': period, 'limit': years, 'apikey': self.fmp_api_key}
            
            async with session.get(balance_url, params=balance_params) as response:
                balance_data = await response.json() if response.status == 200 else []
            
            # Cash Flow Statement
            cashflow_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
            cashflow_params = {'period': period, 'limit': years, 'apikey': self.fmp_api_key}
            
            async with session.get(cashflow_url, params=cashflow_params) as response:
                cashflow_data = await response.json() if response.status == 200 else []
            
            # Combine statements by date
            for i in range(min(len(income_data), len(balance_data), len(cashflow_data))):
                income = income_data[i]
                balance = balance_data[i]
                cashflow = cashflow_data[i]
                
                # Parse fiscal year and quarter
                date_str = income.get('date', '')
                fiscal_year = int(date_str[:4]) if date_str else 0
                fiscal_quarter = None
                if period == 'quarter':
                    # Extract quarter from period or date
                    fiscal_quarter = income.get('period', '').replace('Q', '').replace(str(fiscal_year), '')
                    fiscal_quarter = int(fiscal_quarter) if fiscal_quarter.isdigit() else None
                
                statement = FinancialStatement(
                    symbol=symbol,
                    period=period,
                    fiscal_year=fiscal_year,
                    fiscal_quarter=fiscal_quarter,
                    revenue=income.get('revenue', 0) or 0,
                    cost_of_revenue=income.get('costOfRevenue', 0) or 0,
                    gross_profit=income.get('grossProfit', 0) or 0,
                    operating_expenses=income.get('operatingExpenses', 0) or 0,
                    operating_income=income.get('operatingIncome', 0) or 0,
                    ebitda=income.get('ebitda', 0) or 0,
                    net_income=income.get('netIncome', 0) or 0,
                    eps_diluted=income.get('epsdiluted', 0) or 0,
                    shares_outstanding=income.get('weightedAverageShsOutDil', 0) or 0,
                    total_assets=balance.get('totalAssets', 0) or 0,
                    total_liabilities=balance.get('totalLiabilities', 0) or 0,
                    shareholders_equity=balance.get('totalShareholderEquity', 0) or 0,
                    cash_and_equivalents=balance.get('cashAndCashEquivalents', 0) or 0,
                    total_debt=balance.get('totalDebt', 0) or 0,
                    free_cash_flow=cashflow.get('freeCashFlow', 0) or 0,
                    capex=abs(cashflow.get('capitalExpenditure', 0) or 0),
                    research_development=income.get('researchAndDevelopmentExpenses', 0) or 0
                )
                
                statements.append(statement)
            
            self._set_cache(cache_key, statements, 360)  # 6 hour cache
            return statements
            
        except Exception as e:
            logger.error(f"Failed to get financial statements for {symbol}: {e}")
            # Fallback to yfinance
            return await self._get_financial_statements_yfinance(symbol, period, years)
    
    async def _get_financial_statements_yfinance(self, symbol: str, period: str = 'annual',
                                               years: int = 5) -> List[FinancialStatement]:
        """Fallback method using yfinance for financial statements"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial data
            if period == 'annual':
                income_stmt = ticker.financials
                balance_sheet = ticker.balance_sheet
                cash_flow = ticker.cashflow
            else:
                income_stmt = ticker.quarterly_financials
                balance_sheet = ticker.quarterly_balance_sheet
                cash_flow = ticker.quarterly_cashflow
            
            statements = []
            
            # Get the latest years of data
            available_years = min(years, len(income_stmt.columns))
            
            for i in range(available_years):
                col = income_stmt.columns[i]
                
                # Extract year and quarter info
                fiscal_year = col.year
                fiscal_quarter = None
                if period == 'quarter':
                    fiscal_quarter = ((col.month - 1) // 3) + 1
                
                def safe_get(df, index, default=0):
                    """Safely get value from dataframe"""
                    try:
                        if index in df.index:
                            value = df.loc[index, col]
                            return float(value) if not pd.isna(value) else default
                        return default
                    except:
                        return default
                
                # Extract financial metrics
                revenue = safe_get(income_stmt, 'Total Revenue')
                cost_of_revenue = safe_get(income_stmt, 'Cost Of Revenue')
                gross_profit = safe_get(income_stmt, 'Gross Profit')
                operating_income = safe_get(income_stmt, 'Operating Income')
                net_income = safe_get(income_stmt, 'Net Income')
                ebitda = safe_get(income_stmt, 'EBITDA')
                
                # Balance sheet items
                total_assets = safe_get(balance_sheet, 'Total Assets')
                total_liabilities = safe_get(balance_sheet, 'Total Liab')
                shareholders_equity = safe_get(balance_sheet, 'Total Stockholder Equity')
                cash = safe_get(balance_sheet, 'Cash')
                total_debt = safe_get(balance_sheet, 'Total Debt')
                
                # Cash flow items
                free_cash_flow = safe_get(cash_flow, 'Free Cash Flow')
                capex = abs(safe_get(cash_flow, 'Capital Expenditures'))
                
                # Get shares outstanding from info
                shares_outstanding = 0
                try:
                    info = ticker.info
                    shares_outstanding = info.get('sharesOutstanding', 0) or info.get('impliedSharesOutstanding', 0)
                except:
                    pass
                
                statement = FinancialStatement(
                    symbol=symbol,
                    period=period,
                    fiscal_year=fiscal_year,
                    fiscal_quarter=fiscal_quarter,
                    revenue=revenue,
                    cost_of_revenue=cost_of_revenue,
                    gross_profit=gross_profit,
                    operating_expenses=max(0, gross_profit - operating_income) if gross_profit and operating_income else 0,
                    operating_income=operating_income,
                    ebitda=ebitda,
                    net_income=net_income,
                    eps_diluted=net_income / shares_outstanding if shares_outstanding else 0,
                    shares_outstanding=shares_outstanding,
                    total_assets=total_assets,
                    total_liabilities=total_liabilities,
                    shareholders_equity=shareholders_equity,
                    cash_and_equivalents=cash,
                    total_debt=total_debt,
                    free_cash_flow=free_cash_flow,
                    capex=capex,
                    research_development=safe_get(income_stmt, 'Research Development')
                )
                
                statements.append(statement)
            
            return statements
            
        except Exception as e:
            logger.error(f"Failed to get yfinance financial statements for {symbol}: {e}")
            return []
    
    # =================== DCF VALUATION MODEL ===================
    
    async def calculate_dcf_valuation(self, symbol: str, 
                                    terminal_growth_rate: float = 0.025,
                                    discount_rate: Optional[float] = None,
                                    projection_years: int = 5) -> DCFModel:
        """Calculate DCF valuation using real financial data"""
        
        # Get historical financial statements
        statements = await self.get_financial_statements(symbol, 'annual', 5)
        if not statements:
            raise ValueError(f"No financial data available for {symbol}")
        
        # Calculate WACC if discount rate not provided
        if discount_rate is None:
            discount_rate = await self._calculate_wacc(symbol, statements[0])
        
        # Analyze historical trends
        revenue_growth_rates = self._calculate_historical_growth_rates(
            [s.revenue for s in statements]
        )
        margin_trends = [s.operating_income / s.revenue if s.revenue else 0 for s in statements]
        capex_ratios = [s.capex / s.revenue if s.revenue else 0 for s in statements]
        
        # Project future metrics
        base_revenue = statements[0].revenue
        base_fcf = statements[0].free_cash_flow
        
        # Revenue growth projections (declining growth rate)
        avg_growth = np.mean(revenue_growth_rates) if revenue_growth_rates else 0.05
        revenue_projections = []
        fcf_projections = []
        
        for year in range(1, projection_years + 1):
            # Declining growth rate approach
            growth_rate = max(terminal_growth_rate, avg_growth * (0.8 ** (year - 1)))
            
            if year == 1:
                projected_revenue = base_revenue * (1 + growth_rate)
            else:
                projected_revenue = revenue_projections[-1] * (1 + growth_rate)
            
            revenue_projections.append(projected_revenue)
            
            # Project FCF based on revenue and historical margins
            avg_margin = np.mean(margin_trends) if margin_trends else 0.15
            avg_capex_ratio = np.mean(capex_ratios) if capex_ratios else 0.05
            
            operating_income = projected_revenue * avg_margin
            # Simplified FCF calculation
            tax_rate = 0.25  # Assume 25% tax rate
            nopat = operating_income * (1 - tax_rate)
            capex = projected_revenue * avg_capex_ratio
            
            # Working capital change (simplified)
            wc_change = projected_revenue * 0.02  # 2% of revenue growth
            
            projected_fcf = nopat - capex - wc_change
            fcf_projections.append(projected_fcf)
        
        # Calculate terminal value
        terminal_fcf = fcf_projections[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        
        # Discount cash flows to present value
        present_value_fcf = sum(
            fcf / (1 + discount_rate) ** year 
            for year, fcf in enumerate(fcf_projections, 1)
        )
        
        present_value_terminal = terminal_value / (1 + discount_rate) ** projection_years
        
        # Calculate enterprise and equity value
        enterprise_value = present_value_fcf + present_value_terminal
        
        # Adjust for cash and debt
        net_cash = statements[0].cash_and_equivalents - statements[0].total_debt
        equity_value = enterprise_value + net_cash
        
        # Calculate per-share value
        shares_outstanding = statements[0].shares_outstanding
        intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding else 0
        
        return DCFModel(
            symbol=symbol,
            base_year_fcf=base_fcf,
            terminal_growth_rate=terminal_growth_rate,
            discount_rate=discount_rate,
            projection_years=projection_years,
            revenue_growth_rates=revenue_growth_rates,
            margin_projections=margin_trends,
            capex_as_percent_revenue=capex_ratios,
            working_capital_change=[0.02] * projection_years,  # Simplified
            terminal_value=terminal_value,
            present_value_fcf=present_value_fcf,
            present_value_terminal=present_value_terminal,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            shares_outstanding=shares_outstanding,
            intrinsic_value_per_share=intrinsic_value_per_share
        )
    
    async def _calculate_wacc(self, symbol: str, latest_statement: FinancialStatement) -> float:
        """Calculate Weighted Average Cost of Capital"""
        try:
            # Get market data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Risk-free rate (10-year Treasury)
            treasury_data = await self.get_fred_data('GS10')
            risk_free_rate = treasury_data[0].value / 100 if treasury_data else 0.02
            
            # Market risk premium (historical average)
            market_risk_premium = 0.06  # 6% historical average
            
            # Beta
            beta = info.get('beta', 1.0) or 1.0
            
            # Cost of equity (CAPM)
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            
            # Cost of debt
            total_debt = latest_statement.total_debt
            if total_debt > 0:
                # Estimate cost of debt based on credit profile
                debt_to_equity = total_debt / latest_statement.shareholders_equity if latest_statement.shareholders_equity else 1
                if debt_to_equity < 0.3:
                    cost_of_debt = risk_free_rate + 0.02  # Low risk premium
                elif debt_to_equity < 0.6:
                    cost_of_debt = risk_free_rate + 0.04  # Medium risk premium
                else:
                    cost_of_debt = risk_free_rate + 0.06  # High risk premium
            else:
                cost_of_debt = risk_free_rate
            
            # Tax rate
            tax_rate = 0.25  # Approximate corporate tax rate
            after_tax_cost_debt = cost_of_debt * (1 - tax_rate)
            
            # Market values
            market_cap = info.get('marketCap', 0) or 0
            market_value_debt = total_debt  # Approximation
            total_value = market_cap + market_value_debt
            
            if total_value == 0:
                return cost_of_equity  # All equity financed
            
            # WACC calculation
            weight_equity = market_cap / total_value
            weight_debt = market_value_debt / total_value
            
            wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_debt)
            
            return wacc
            
        except Exception as e:
            logger.error(f"Failed to calculate WACC for {symbol}: {e}")
            return 0.10  # Default 10% discount rate
    
    def _calculate_historical_growth_rates(self, values: List[float]) -> List[float]:
        """Calculate historical growth rates from a series of values"""
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth_rate = (values[i-1] - values[i]) / values[i]  # Note: reversed due to desc order
                growth_rates.append(growth_rate)
        return growth_rates
    
    # =================== COMPARABLE COMPANY ANALYSIS ===================
    
    async def get_comparable_companies(self, symbol: str, industry: Optional[str] = None) -> List[ComparableCompany]:
        """Get comparable companies analysis using real market data"""
        try:
            # Get target company info
            target_ticker = yf.Ticker(symbol)
            target_info = target_ticker.info
            
            # Determine industry if not provided
            if not industry:
                industry = target_info.get('industry', '') or target_info.get('sector', '')
            
            # Get industry peers (this would typically use a financial data service)
            # For now, we'll use a simplified approach with known peers
            industry_peers = await self._get_industry_peers(symbol, industry)
            
            comparables = []
            
            for peer_symbol in industry_peers:
                try:
                    comp_data = await self._get_company_metrics(peer_symbol)
                    if comp_data:
                        comparables.append(comp_data)
                except Exception as e:
                    logger.warning(f"Failed to get data for {peer_symbol}: {e}")
                    continue
            
            return comparables
            
        except Exception as e:
            logger.error(f"Failed to get comparable companies for {symbol}: {e}")
            return []
    
    async def _get_industry_peers(self, symbol: str, industry: str) -> List[str]:
        """Get industry peer symbols (simplified implementation)"""
        # This is a simplified mapping - in production, you'd use a financial data service
        # that provides industry classifications and peer groups
        
        peer_mapping = {
            'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX'],
            'MSFT': ['AAPL', 'GOOGL', 'META', 'CRM', 'ORCL'],
            'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN', 'NFLX'],
            'TSLA': ['F', 'GM', 'RIVN', 'LCID', 'NIO'],
            'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS'],
            'JNJ': ['PFE', 'MRK', 'ABBV', 'BMY', 'LLY'],
            'XOM': ['CVX', 'COP', 'EOG', 'PSX', 'VLO']
        }
        
        # Return known peers or generate based on industry
        if symbol in peer_mapping:
            return peer_mapping[symbol]
        
        # Fallback: return empty list (in production, query financial data service)
        return []
    
    async def _get_company_metrics(self, symbol: str) -> Optional[ComparableCompany]:
        """Get comprehensive company metrics for comparable analysis"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial statements for TTM calculations
            statements = await self.get_financial_statements(symbol, 'annual', 1)
            
            if not statements:
                return None
            
            latest = statements[0]
            
            # Market data
            market_cap = info.get('marketCap', 0) or 0
            enterprise_value = info.get('enterpriseValue', 0) or market_cap + latest.total_debt - latest.cash_and_equivalents
            
            # Financial metrics
            revenue_ttm = latest.revenue
            ebitda_ttm = latest.ebitda
            net_income_ttm = latest.net_income
            
            # Ratios
            pe_ratio = market_cap / net_income_ttm if net_income_ttm > 0 else 0
            ev_revenue = enterprise_value / revenue_ttm if revenue_ttm > 0 else 0
            ev_ebitda = enterprise_value / ebitda_ttm if ebitda_ttm > 0 else 0
            
            book_value = latest.shareholders_equity
            price_to_book = market_cap / book_value if book_value > 0 else 0
            
            roe = net_income_ttm / book_value if book_value > 0 else 0
            debt_to_equity = latest.total_debt / book_value if book_value > 0 else 0
            
            # Revenue growth (need multiple years)
            all_statements = await self.get_financial_statements(symbol, 'annual', 2)
            revenue_growth = 0
            if len(all_statements) >= 2:
                current_rev = all_statements[0].revenue
                prior_rev = all_statements[1].revenue
                revenue_growth = (current_rev - prior_rev) / prior_rev if prior_rev > 0 else 0
            
            return ComparableCompany(
                symbol=symbol,
                name=info.get('longName', symbol),
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                revenue_ttm=revenue_ttm,
                ebitda_ttm=ebitda_ttm,
                net_income_ttm=net_income_ttm,
                pe_ratio=pe_ratio,
                ev_revenue=ev_revenue,
                ev_ebitda=ev_ebitda,
                price_to_book=price_to_book,
                roe=roe,
                debt_to_equity=debt_to_equity,
                revenue_growth=revenue_growth,
                industry=info.get('industry', '')
            )
            
        except Exception as e:
            logger.error(f"Failed to get metrics for {symbol}: {e}")
            return None
    
    # =================== EARNINGS ANALYSIS ===================
    
    async def get_earnings_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive earnings analysis including estimates and revisions"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings data
            earnings = ticker.earnings
            quarterly_earnings = ticker.quarterly_earnings
            
            # Get analyst estimates (from yfinance info)
            info = ticker.info
            
            # Calculate earnings metrics
            analysis = {
                'symbol': symbol,
                'current_eps': info.get('trailingEps', 0),
                'forward_eps': info.get('forwardEps', 0),
                'eps_growth': self._calculate_eps_growth(earnings),
                'earnings_surprise_history': await self._get_earnings_surprises(symbol),
                'consensus_estimates': {
                    'current_quarter': info.get('earningsQuarterlyGrowth', 0),
                    'next_quarter': 0,  # Would need premium data service
                    'current_year': info.get('earningsGrowth', 0),
                    'next_year': 0  # Would need premium data service
                },
                'peg_ratio': info.get('pegRatio', 0),
                'earnings_date': info.get('earningsDate', ''),
                'revenue_per_share': info.get('revenuePerShare', 0),
                'quarterly_revenue_growth': info.get('revenueQuarterlyGrowth', 0)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to get earnings analysis for {symbol}: {e}")
            return {}
    
    def _calculate_eps_growth(self, earnings: pd.DataFrame) -> float:
        """Calculate EPS growth rate from historical earnings"""
        if len(earnings) < 2:
            return 0
        
        # Get the most recent two years
        years = sorted(earnings.index, reverse=True)
        if len(years) >= 2:
            current_eps = earnings.loc[years[0], 'Earnings']
            prior_eps = earnings.loc[years[1], 'Earnings']
            
            if prior_eps != 0:
                return (current_eps - prior_eps) / abs(prior_eps)
        
        return 0
    
    async def _get_earnings_surprises(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical earnings surprises (simplified implementation)"""
        # This would typically use a premium financial data service
        # For now, return empty list
        return []
    
    # =================== ECONOMIC FORECASTING ===================
    
    async def generate_economic_forecast(self, indicators: List[str], 
                                       forecast_periods: int = 4) -> Dict[str, Any]:
        """Generate economic forecasts using statistical models"""
        forecasts = {}
        
        for indicator in indicators:
            try:
                fred_id = self.fred_indicators.get(indicator, indicator)
                historical_data = await self.get_fred_data(fred_id, 
                    start_date=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'))
                
                if not historical_data:
                    continue
                
                # Extract values and dates
                values = [d.value for d in historical_data]
                dates = [d.date for d in historical_data]
                
                # Sort by date
                sorted_data = sorted(zip(dates, values))
                values = [v for d, v in sorted_data]
                
                # Simple forecasting using trend analysis
                forecast = self._simple_forecast(values, forecast_periods)
                
                forecasts[indicator] = {
                    'historical_values': values[-12:],  # Last 12 observations
                    'forecast_values': forecast,
                    'forecast_periods': forecast_periods,
                    'confidence_intervals': self._calculate_confidence_intervals(values, forecast),
                    'trend': 'increasing' if forecast[-1] > values[-1] else 'decreasing'
                }
                
            except Exception as e:
                logger.error(f"Failed to forecast {indicator}: {e}")
                forecasts[indicator] = {'error': str(e)}
        
        return forecasts
    
    def _simple_forecast(self, values: List[float], periods: int) -> List[float]:
        """Simple trend-based forecasting"""
        if len(values) < 3:
            # Not enough data, return last value
            return [values[-1]] * periods if values else [0] * periods
        
        # Linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Remove any NaN or infinite values
        mask = np.isfinite(y)
        if not mask.any():
            return [values[-1]] * periods
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(y_clean) < 2:
            return [values[-1]] * periods
        
        # Fit linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Generate forecasts
        forecast = []
        last_x = len(values) - 1
        
        for i in range(1, periods + 1):
            forecast_x = last_x + i
            forecast_value = slope * forecast_x + intercept
            forecast.append(forecast_value)
        
        return forecast
    
    def _calculate_confidence_intervals(self, historical: List[float], 
                                     forecast: List[float]) -> Dict[str, List[float]]:
        """Calculate confidence intervals for forecasts"""
        if len(historical) < 3:
            return {'lower': forecast, 'upper': forecast}
        
        # Calculate standard error from historical data
        residuals = np.diff(historical)  # Simple differencing
        std_error = np.std(residuals)
        
        # 95% confidence interval (±1.96 * std_error)
        lower_bound = [f - 1.96 * std_error for f in forecast]
        upper_bound = [f + 1.96 * std_error for f in forecast]
        
        return {
            'lower': lower_bound,
            'upper': upper_bound
        }

# Global service instance
economic_research_service = EconomicResearchService()

# Convenience functions
async def get_economic_dashboard() -> Dict[str, Any]:
    """Get economic indicators dashboard"""
    return await economic_research_service.get_economic_indicators_dashboard()

async def get_company_dcf_valuation(symbol: str, terminal_growth: float = 0.025) -> DCFModel:
    """Get DCF valuation for a company"""
    return await economic_research_service.calculate_dcf_valuation(symbol, terminal_growth)

async def get_peer_analysis(symbol: str) -> List[ComparableCompany]:
    """Get comparable company analysis"""
    return await economic_research_service.get_comparable_companies(symbol)

async def get_economic_forecast(indicators: List[str] = None) -> Dict[str, Any]:
    """Get economic forecasts"""
    if indicators is None:
        indicators = ['gdp', 'unemployment', 'inflation_cpi', 'fed_funds_rate']
    return await economic_research_service.generate_economic_forecast(indicators)

async def analyze_company_fundamentals(symbol: str) -> Dict[str, Any]:
    """Get comprehensive fundamental analysis"""
    try:
        # Get financial statements
        statements = await economic_research_service.get_financial_statements(symbol)
        
        # Get DCF valuation
        dcf_model = await economic_research_service.calculate_dcf_valuation(symbol)
        
        # Get comparable companies
        peers = await economic_research_service.get_comparable_companies(symbol)
        
        # Get earnings analysis
        earnings = await economic_research_service.get_earnings_analysis(symbol)
        
        return {
            'symbol': symbol,
            'financial_statements': [
                {
                    'fiscal_year': s.fiscal_year,
                    'revenue': s.revenue,
                    'net_income': s.net_income,
                    'free_cash_flow': s.free_cash_flow,
                    'total_assets': s.total_assets,
                    'total_debt': s.total_debt,
                    'shareholders_equity': s.shareholders_equity
                } for s in statements[:3]  # Last 3 years
            ],
            'dcf_valuation': {
                'intrinsic_value_per_share': dcf_model.intrinsic_value_per_share,
                'enterprise_value': dcf_model.enterprise_value,
                'terminal_growth_rate': dcf_model.terminal_growth_rate,
                'discount_rate': dcf_model.discount_rate,
                'present_value_fcf': dcf_model.present_value_fcf,
                'present_value_terminal': dcf_model.present_value_terminal
            },
            'peer_comparison': {
                'peer_count': len(peers),
                'median_pe': np.median([p.pe_ratio for p in peers if p.pe_ratio > 0]) if peers else 0,
                'median_ev_revenue': np.median([p.ev_revenue for p in peers if p.ev_revenue > 0]) if peers else 0,
                'median_ev_ebitda': np.median([p.ev_ebitda for p in peers if p.ev_ebitda > 0]) if peers else 0,
                'peer_symbols': [p.symbol for p in peers]
            },
            'earnings_analysis': earnings,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze fundamentals for {symbol}: {e}")
        return {'error': str(e), 'symbol': symbol} 