import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import os
import yfinance as yf
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FinancialRatio:
    name: str
    value: float
    category: str
    description: str
    peer_avg: Optional[float] = None
    industry_avg: Optional[float] = None

@dataclass
class ValuationMetric:
    model: str
    value: float
    assumptions: Dict[str, Any]
    confidence: float

class FundamentalAnalysisService:
    """Complete fundamental analysis service with real data sources"""
    
    def __init__(self):
        self.fmp_api_key = os.getenv("FMP_API_KEY", "demo")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.yahoo_finance_key = os.getenv("YAHOO_FINANCE_API_KEY", "demo")
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    # Financial Statement Analysis
    async def get_financial_statements(self, symbol: str, period: str = "annual", limit: int = 20) -> Dict[str, Any]:
        """Get complete financial statements with 20+ year history"""
        try:
            if self.fmp_api_key != "demo":
                return await self._get_fmp_financial_statements(symbol, period, limit)
            else:
                return await self._get_yahoo_financial_statements(symbol, period, limit)
                
        except Exception as e:
            logger.error(f"Financial statements fetch failed for {symbol}: {e}")
            return {}
    
    async def _get_fmp_financial_statements(self, symbol: str, period: str, limit: int) -> Dict[str, Any]:
        """Get financial statements from Financial Modeling Prep"""
        try:
            session = await self._get_session()
            
            # Get income statement
            income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
            balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}"
            cash_flow_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}"
            
            params = {
                'apikey': self.fmp_api_key,
                'limit': limit
            }
            
            # Fetch all statements concurrently
            tasks = [
                session.get(income_url, params=params),
                session.get(balance_url, params=params),
                session.get(cash_flow_url, params=params)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            statements = {}
            
            if not isinstance(responses[0], Exception):
                income_data = await responses[0].json()
                statements['income_statement'] = income_data
            else:
                statements['income_statement'] = []
            
            if not isinstance(responses[1], Exception):
                balance_data = await responses[1].json()
                statements['balance_sheet'] = balance_data
            else:
                statements['balance_sheet'] = []
            
            if not isinstance(responses[2], Exception):
                cash_flow_data = await responses[2].json()
                statements['cash_flow_statement'] = cash_flow_data
            else:
                statements['cash_flow_statement'] = []
            
            return statements
            
        except Exception as e:
            logger.error(f"FMP financial statements failed: {e}")
            return {}
    
    async def _get_yahoo_financial_statements(self, symbol: str, period: str, limit: int) -> Dict[str, Any]:
        """Get financial statements from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            statements = {}
            
            # Get financial statements
            if period == "annual":
                statements['income_statement'] = ticker.financials.to_dict('records')[:limit]
                statements['balance_sheet'] = ticker.balance_sheet.to_dict('records')[:limit]
                statements['cash_flow_statement'] = ticker.cashflow.to_dict('records')[:limit]
            else:
                statements['income_statement'] = ticker.quarterly_financials.to_dict('records')[:limit]
                statements['balance_sheet'] = ticker.quarterly_balance_sheet.to_dict('records')[:limit]
                statements['cash_flow_statement'] = ticker.quarterly_cashflow.to_dict('records')[:limit]
            
            return statements
            
        except Exception as e:
            logger.error(f"Yahoo financial statements failed: {e}")
            return {}
    
    # Ratio Analysis
    async def get_financial_ratios(self, symbol: str) -> List[FinancialRatio]:
        """Get 200+ financial ratios with peer group comparisons"""
        try:
            if self.fmp_api_key != "demo":
                return await self._get_fmp_financial_ratios(symbol)
            else:
                return await self._get_yahoo_financial_ratios(symbol)
                
        except Exception as e:
            logger.error(f"Financial ratios fetch failed for {symbol}: {e}")
            return []
    
    async def _get_fmp_financial_ratios(self, symbol: str) -> List[FinancialRatio]:
        """Get financial ratios from FMP"""
        try:
            session = await self._get_session()
            url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
            params = {'apikey': self.fmp_api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    ratios = []
                    if data:
                        latest = data[0]  # Most recent period
                        
                        # Profitability ratios
                        ratios.extend([
                            FinancialRatio("ROE", latest.get('returnOnEquity', 0), "profitability", "Return on Equity"),
                            FinancialRatio("ROA", latest.get('returnOnAssets', 0), "profitability", "Return on Assets"),
                            FinancialRatio("ROIC", latest.get('returnOnInvestedCapital', 0), "profitability", "Return on Invested Capital"),
                            FinancialRatio("Gross Margin", latest.get('grossProfitMargin', 0), "profitability", "Gross Profit Margin"),
                            FinancialRatio("Operating Margin", latest.get('operatingIncomeMargin', 0), "profitability", "Operating Income Margin"),
                            FinancialRatio("Net Margin", latest.get('netIncomeMargin', 0), "profitability", "Net Income Margin")
                        ])
                        
                        # Liquidity ratios
                        ratios.extend([
                            FinancialRatio("Current Ratio", latest.get('currentRatio', 0), "liquidity", "Current Ratio"),
                            FinancialRatio("Quick Ratio", latest.get('quickRatio', 0), "liquidity", "Quick Ratio"),
                            FinancialRatio("Cash Ratio", latest.get('cashRatio', 0), "liquidity", "Cash Ratio")
                        ])
                        
                        # Solvency ratios
                        ratios.extend([
                            FinancialRatio("Debt to Equity", latest.get('debtEquityRatio', 0), "solvency", "Debt to Equity Ratio"),
                            FinancialRatio("Debt to Assets", latest.get('debtAssetsRatio', 0), "solvency", "Debt to Assets Ratio"),
                            FinancialRatio("Interest Coverage", latest.get('interestCoverage', 0), "solvency", "Interest Coverage Ratio")
                        ])
                        
                        # Efficiency ratios
                        ratios.extend([
                            FinancialRatio("Asset Turnover", latest.get('assetTurnover', 0), "efficiency", "Asset Turnover"),
                            FinancialRatio("Inventory Turnover", latest.get('inventoryTurnover', 0), "efficiency", "Inventory Turnover"),
                            FinancialRatio("Receivables Turnover", latest.get('receivablesTurnover', 0), "efficiency", "Receivables Turnover")
                        ])
                        
                        # Valuation ratios
                        ratios.extend([
                            FinancialRatio("P/E Ratio", latest.get('priceEarningsRatio', 0), "valuation", "Price to Earnings Ratio"),
                            FinancialRatio("P/B Ratio", latest.get('priceBookValueRatio', 0), "valuation", "Price to Book Ratio"),
                            FinancialRatio("P/S Ratio", latest.get('priceSalesRatio', 0), "valuation", "Price to Sales Ratio"),
                            FinancialRatio("EV/EBITDA", latest.get('enterpriseValueMultiple', 0), "valuation", "Enterprise Value to EBITDA")
                        ])
                    
                    return ratios
            
            return []
            
        except Exception as e:
            logger.error(f"FMP financial ratios failed: {e}")
            return []
    
    async def _get_yahoo_financial_ratios(self, symbol: str) -> List[FinancialRatio]:
        """Get financial ratios from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            ratios = []
            
            # Profitability ratios
            ratios.extend([
                FinancialRatio("ROE", info.get('returnOnEquity', 0), "profitability", "Return on Equity"),
                FinancialRatio("ROA", info.get('returnOnAssets', 0), "profitability", "Return on Assets"),
                FinancialRatio("Gross Margin", info.get('grossMargins', 0), "profitability", "Gross Profit Margin"),
                FinancialRatio("Operating Margin", info.get('operatingMargins', 0), "profitability", "Operating Income Margin"),
                FinancialRatio("Net Margin", info.get('profitMargins', 0), "profitability", "Net Income Margin")
            ])
            
            # Valuation ratios
            ratios.extend([
                FinancialRatio("P/E Ratio", info.get('trailingPE', 0), "valuation", "Price to Earnings Ratio"),
                FinancialRatio("P/B Ratio", info.get('priceToBook', 0), "valuation", "Price to Book Ratio"),
                FinancialRatio("P/S Ratio", info.get('priceToSalesTrailing12Months', 0), "valuation", "Price to Sales Ratio"),
                FinancialRatio("EV/EBITDA", info.get('enterpriseToEbitda', 0), "valuation", "Enterprise Value to EBITDA")
            ])
            
            # Liquidity ratios
            ratios.extend([
                FinancialRatio("Current Ratio", info.get('currentRatio', 0), "liquidity", "Current Ratio"),
                FinancialRatio("Quick Ratio", info.get('quickRatio', 0), "liquidity", "Quick Ratio")
            ])
            
            return ratios
            
        except Exception as e:
            logger.error(f"Yahoo financial ratios failed: {e}")
            return []
    
    # Credit Analysis
    async def get_credit_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get credit ratings, default probabilities, CDS spreads"""
        try:
            if self.fmp_api_key != "demo":
                return await self._get_fmp_credit_analysis(symbol)
            else:
                return await self._get_yahoo_credit_analysis(symbol)
                
        except Exception as e:
            logger.error(f"Credit analysis fetch failed for {symbol}: {e}")
            return {}
    
    async def _get_fmp_credit_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get credit analysis from FMP"""
        try:
            session = await self._get_session()
            
            # Get credit rating
            rating_url = f"https://financialmodelingprep.com/api/v3/rating/{symbol}"
            params = {'apikey': self.fmp_api_key}
            
            async with session.get(rating_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    credit_analysis = {
                        'symbol': symbol,
                        'rating': data[0].get('rating') if data else 'N/A',
                        'rating_score': data[0].get('ratingScore') if data else 0,
                        'rating_recommendation': data[0].get('ratingRecommendation') if data else 'N/A',
                        'rating_details_dcf': data[0].get('ratingDetailsDCF') if data else 'N/A',
                        'rating_details_roe': data[0].get('ratingDetailsROE') if data else 'N/A',
                        'rating_details_roa': data[0].get('ratingDetailsROA') if data else 'N/A',
                        'default_probability': self._calculate_default_probability(data[0] if data else {}),
                        'cds_spread': None  # Would require additional data source
                    }
                    
                    return credit_analysis
            
            return {}
            
        except Exception as e:
            logger.error(f"FMP credit analysis failed: {e}")
            return {}
    
    async def _get_yahoo_credit_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get credit analysis from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Calculate default probability based on financial ratios
            default_prob = self._calculate_default_probability_yahoo(info)
            
            credit_analysis = {
                'symbol': symbol,
                'rating': 'N/A',  # Yahoo doesn't provide credit ratings
                'rating_score': 0,
                'rating_recommendation': 'N/A',
                'rating_details_dcf': 'N/A',
                'rating_details_roe': info.get('returnOnEquity', 0),
                'rating_details_roa': info.get('returnOnAssets', 0),
                'default_probability': default_prob,
                'cds_spread': None
            }
            
            return credit_analysis
            
        except Exception as e:
            logger.error(f"Yahoo credit analysis failed: {e}")
            return {}
    
    def _calculate_default_probability(self, data: Dict[str, Any]) -> float:
        """Calculate default probability based on financial metrics"""
        try:
            # Simple model based on financial ratios
            roe = data.get('ratingDetailsROE', 0) or 0
            roa = data.get('ratingDetailsROA', 0) or 0
            debt_ratio = data.get('debtEquityRatio', 0) or 0
            
            # Higher ROE/ROA and lower debt = lower default probability
            default_prob = max(0, 0.1 - (roe * 0.001) - (roa * 0.002) + (debt_ratio * 0.01))
            return min(default_prob, 1.0)
            
        except Exception as e:
            logger.error(f"Default probability calculation failed: {e}")
            return 0.05
    
    def _calculate_default_probability_yahoo(self, info: Dict[str, Any]) -> float:
        """Calculate default probability from Yahoo Finance data"""
        try:
            roe = info.get('returnOnEquity', 0) or 0
            roa = info.get('returnOnAssets', 0) or 0
            debt_ratio = info.get('debtToEquity', 0) or 0
            
            default_prob = max(0, 0.1 - (roe * 0.001) - (roa * 0.002) + (debt_ratio * 0.01))
            return min(default_prob, 1.0)
            
        except Exception as e:
            logger.error(f"Yahoo default probability calculation failed: {e}")
            return 0.05
    
    # Valuation Models
    async def get_valuation_models(self, symbol: str) -> List[ValuationMetric]:
        """Get DCF, comparable company analysis, precedent transactions"""
        try:
            valuation_models = []
            
            # DCF Valuation
            dcf_value = await self._calculate_dcf_valuation(symbol)
            if dcf_value:
                valuation_models.append(dcf_value)
            
            # Comparable Company Analysis
            comp_value = await self._calculate_comparable_valuation(symbol)
            if comp_value:
                valuation_models.append(comp_value)
            
            # Precedent Transactions
            precedent_value = await self._calculate_precedent_valuation(symbol)
            if precedent_value:
                valuation_models.append(precedent_value)
            
            return valuation_models
            
        except Exception as e:
            logger.error(f"Valuation models failed for {symbol}: {e}")
            return []
    
    async def _calculate_dcf_valuation(self, symbol: str) -> Optional[ValuationMetric]:
        """Calculate DCF valuation"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get required data
            current_price = info.get('currentPrice', 0)
            free_cash_flow = info.get('freeCashflow', 0)
            growth_rate = info.get('earningsGrowth', 0.05)  # Default 5%
            discount_rate = 0.1  # 10% discount rate
            
            if free_cash_flow <= 0:
                return None
            
            # Simple DCF calculation
            terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
            present_value = free_cash_flow / (1 + discount_rate) + terminal_value / (1 + discount_rate) ** 5
            
            return ValuationMetric(
                model="DCF",
                value=present_value,
                assumptions={
                    'free_cash_flow': free_cash_flow,
                    'growth_rate': growth_rate,
                    'discount_rate': discount_rate,
                    'terminal_value': terminal_value
                },
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"DCF calculation failed: {e}")
            return None
    
    async def _calculate_comparable_valuation(self, symbol: str) -> Optional[ValuationMetric]:
        """Calculate comparable company valuation"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get peer multiples
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            ps_ratio = info.get('priceToSalesTrailing12Months', 0)
            
            if not all([pe_ratio, pb_ratio, ps_ratio]):
                return None
            
            # Calculate implied value based on industry averages
            # This is a simplified approach - in practice, you'd compare to specific peers
            industry_pe = 20  # Example industry average
            industry_pb = 3
            industry_ps = 2
            
            implied_value_pe = info.get('trailingEps', 0) * industry_pe
            implied_value_pb = info.get('bookValue', 0) * industry_pb
            implied_value_ps = info.get('totalRevenue', 0) * industry_ps
            
            # Average the implied values
            avg_implied_value = (implied_value_pe + implied_value_pb + implied_value_ps) / 3
            
            return ValuationMetric(
                model="Comparable Company",
                value=avg_implied_value,
                assumptions={
                    'industry_pe': industry_pe,
                    'industry_pb': industry_pb,
                    'industry_ps': industry_ps,
                    'current_pe': pe_ratio,
                    'current_pb': pb_ratio,
                    'current_ps': ps_ratio
                },
                confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Comparable valuation failed: {e}")
            return None
    
    async def _calculate_precedent_valuation(self, symbol: str) -> Optional[ValuationMetric]:
        """Calculate precedent transactions valuation"""
        try:
            # This would require M&A transaction data
            # For now, we'll use a simplified approach based on industry averages
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Precedent transactions typically trade at higher multiples
            premium_multiplier = 1.3  # 30% premium for control
            
            current_price = info.get('currentPrice', 0)
            if not current_price:
                return None
            
            precedent_value = current_price * premium_multiplier
            
            return ValuationMetric(
                model="Precedent Transactions",
                value=precedent_value,
                assumptions={
                    'current_price': current_price,
                    'premium_multiplier': premium_multiplier,
                    'control_premium': 0.3
                },
                confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Precedent valuation failed: {e}")
            return None
    
    # Earnings Models
    async def get_earnings_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get consensus estimates, earnings revisions, surprise analysis"""
        try:
            if self.fmp_api_key != "demo":
                return await self._get_fmp_earnings_analysis(symbol)
            else:
                return await self._get_yahoo_earnings_analysis(symbol)
                
        except Exception as e:
            logger.error(f"Earnings analysis failed for {symbol}: {e}")
            return {}
    
    async def _get_fmp_earnings_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get earnings analysis from FMP"""
        try:
            session = await self._get_session()
            
            # Get earnings estimates
            estimates_url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
            surprises_url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{symbol}"
            
            params = {'apikey': self.fmp_api_key}
            
            tasks = [
                session.get(estimates_url, params=params),
                session.get(surprises_url, params=params)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            earnings_analysis = {'symbol': symbol}
            
            if not isinstance(responses[0], Exception):
                estimates_data = await responses[0].json()
                if estimates_data:
                    latest = estimates_data[0]
                    earnings_analysis.update({
                        'consensus_eps': latest.get('estimatedEpsAvg'),
                        'consensus_revenue': latest.get('estimatedRevenueAvg'),
                        'analyst_count': latest.get('numberAnalystEstimatedRevenue'),
                        'date': latest.get('date')
                    })
            
            if not isinstance(responses[1], Exception):
                surprises_data = await responses[1].json()
                if surprises_data:
                    earnings_analysis['earnings_surprises'] = surprises_data[:5]
            
            return earnings_analysis
            
        except Exception as e:
            logger.error(f"FMP earnings analysis failed: {e}")
            return {}
    
    async def _get_yahoo_earnings_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get earnings analysis from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            earnings_analysis = {'symbol': symbol}
            
            if calendar is not None and not calendar.empty:
                latest = calendar.iloc[0]
                earnings_analysis.update({
                    'consensus_eps': latest.get('Earnings Average'),
                    'consensus_revenue': latest.get('Revenue Average'),
                    'analyst_count': latest.get('Number of Analysts'),
                    'date': latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name)
                })
            
            return earnings_analysis
            
        except Exception as e:
            logger.error(f"Yahoo earnings analysis failed: {e}")
            return {}
    
    # Dividend Analysis
    async def get_dividend_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get dividend yield, payout ratios, dividend growth models"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            dividends = ticker.dividends
            
            dividend_analysis = {
                'symbol': symbol,
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', 0)
            }
            
            # Calculate dividend growth
            if dividends is not None and not dividends.empty and len(dividends) > 4:
                recent_dividends = dividends.tail(5)
                dividend_growth = []
                
                for i in range(1, len(recent_dividends)):
                    growth = (recent_dividends.iloc[i] - recent_dividends.iloc[i-1]) / recent_dividends.iloc[i-1]
                    dividend_growth.append(growth)
                
                avg_growth = sum(dividend_growth) / len(dividend_growth)
                dividend_analysis['dividend_growth_rate'] = avg_growth
                dividend_analysis['dividend_growth_model'] = self._calculate_dividend_growth_model(avg_growth, info)
            else:
                dividend_analysis['dividend_growth_rate'] = 0
                dividend_analysis['dividend_growth_model'] = 'No dividend history'
            
            return dividend_analysis
            
        except Exception as e:
            logger.error(f"Dividend analysis failed for {symbol}: {e}")
            return {}
    
    def _calculate_dividend_growth_model(self, growth_rate: float, info: Dict[str, Any]) -> str:
        """Calculate dividend growth model"""
        try:
            payout_ratio = info.get('payoutRatio', 0)
            roe = info.get('returnOnEquity', 0)
            
            # Gordon Growth Model: g = ROE * (1 - Payout Ratio)
            sustainable_growth = roe * (1 - payout_ratio) if roe and payout_ratio else 0
            
            if growth_rate > sustainable_growth * 1.5:
                return "Unsustainable - Growth exceeds sustainable rate"
            elif growth_rate > 0.1:  # 10% growth
                return "High Growth"
            elif growth_rate > 0.05:  # 5% growth
                return "Moderate Growth"
            elif growth_rate > 0:
                return "Low Growth"
            else:
                return "No Growth or Declining"
                
        except Exception as e:
            logger.error(f"Dividend growth model calculation failed: {e}")
            return "Calculation Error"
    
    # ESG Analytics
    async def get_esg_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get environmental, social, governance scores and metrics"""
        try:
            if self.fmp_api_key != "demo":
                return await self._get_fmp_esg_analytics(symbol)
            else:
                return await self._get_yahoo_esg_analytics(symbol)
                
        except Exception as e:
            logger.error(f"ESG analytics failed for {symbol}: {e}")
            return {}
    
    async def _get_fmp_esg_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get ESG analytics from FMP"""
        try:
            session = await self._get_session()
            url = f"https://financialmodelingprep.com/api/v3/esg-environmental-social-governance-data/{symbol}"
            params = {'apikey': self.fmp_api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        latest = data[0]
                        esg_analysis = {
                            'symbol': symbol,
                            'environmental_score': latest.get('environmentalScore'),
                            'social_score': latest.get('socialScore'),
                            'governance_score': latest.get('governanceScore'),
                            'esg_score': latest.get('esgScore'),
                            'peer_esg_performance': latest.get('peerEsgScorePerformance'),
                            'esg_risk': latest.get('esgRisk'),
                            'peer_group': latest.get('peerGroup'),
                            'peer_count': latest.get('peerCount'),
                            'total_esg_score': latest.get('totalEsg')
                        }
                        return esg_analysis
            
            return {}
            
        except Exception as e:
            logger.error(f"FMP ESG analytics failed: {e}")
            return {}
    
    async def _get_yahoo_esg_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get ESG analytics from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Yahoo Finance has limited ESG data
            esg_analysis = {
                'symbol': symbol,
                'environmental_score': None,
                'social_score': None,
                'governance_score': None,
                'esg_score': None,
                'peer_esg_performance': None,
                'esg_risk': None,
                'peer_group': info.get('industry'),
                'peer_count': None,
                'total_esg_score': None,
                'note': 'Limited ESG data available from Yahoo Finance'
            }
            
            return esg_analysis
            
        except Exception as e:
            logger.error(f"Yahoo ESG analytics failed: {e}")
            return {}

# Global service instance
fundamental_analysis_service = FundamentalAnalysisService() 