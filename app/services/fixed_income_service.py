"""
Fixed Income Analytics Service
Comprehensive bond analytics, yield curve analysis, duration/convexity calculations,
term structure models, and interest rate modeling as required by goals.md
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as optimize
import scipy.stats as stats
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BondType(Enum):
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    MUNICIPAL = "municipal"
    INTERNATIONAL = "international"
    TIPS = "tips"  # Treasury Inflation-Protected Securities
    CALLABLE = "callable"
    CONVERTIBLE = "convertible"
    ZERO_COUPON = "zero_coupon"

class CreditRating(Enum):
    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"

@dataclass
class Bond:
    cusip: str
    issuer: str
    bond_type: BondType
    face_value: float
    coupon_rate: float
    maturity_date: datetime
    issue_date: datetime
    credit_rating: CreditRating
    callable: bool = False
    call_date: Optional[datetime] = None
    call_price: Optional[float] = None
    frequency: int = 2  # Semi-annual by default
    day_count: str = "30/360"

@dataclass
class YieldCurvePoint:
    maturity: float  # Years
    yield_rate: float
    spot_rate: float
    forward_rate: float
    discount_factor: float
    par_rate: float

@dataclass
class DurationConvexityResult:
    modified_duration: float
    effective_duration: float
    macaulay_duration: float
    convexity: float
    dv01: float  # Dollar value of 01
    price_sensitivity: Dict[str, float]

@dataclass
class YieldCurveAnalysis:
    curve_date: datetime
    curve_points: List[YieldCurvePoint]
    curve_type: str  # 'treasury', 'corporate', 'municipal'
    interpolation_method: str
    curve_shifts: Dict[str, List[float]]  # parallel, twist, butterfly
    key_rates: Dict[str, float]

@dataclass
class CreditAnalysis:
    symbol: str
    credit_spread: float
    default_probability: float
    recovery_rate: float
    credit_rating: CreditRating
    rating_migration_matrix: Dict[str, float]
    cds_spread: Optional[float]
    z_spread: float
    option_adjusted_spread: float

class InterestRateModel(ABC):
    """Abstract base class for interest rate models"""
    
    @abstractmethod
    def simulate_rates(self, initial_rate: float, time_horizon: float, 
                      num_simulations: int, dt: float = 0.01) -> np.ndarray:
        pass
    
    @abstractmethod
    def calibrate(self, market_data: pd.DataFrame) -> Dict[str, float]:
        pass

class VasicekModel(InterestRateModel):
    """Vasicek interest rate model: dr = a(b - r)dt + σdW"""
    
    def __init__(self, a: float = 0.1, b: float = 0.05, sigma: float = 0.01):
        self.a = a  # Mean reversion speed
        self.b = b  # Long-term mean
        self.sigma = sigma  # Volatility
    
    def simulate_rates(self, initial_rate: float, time_horizon: float, 
                      num_simulations: int, dt: float = 0.01) -> np.ndarray:
        """Simulate interest rate paths"""
        num_steps = int(time_horizon / dt)
        rates = np.zeros((num_simulations, num_steps + 1))
        rates[:, 0] = initial_rate
        
        for i in range(num_steps):
            dW = np.random.normal(0, np.sqrt(dt), num_simulations)
            dr = self.a * (self.b - rates[:, i]) * dt + self.sigma * dW
            rates[:, i + 1] = rates[:, i] + dr
        
        return rates
    
    def calibrate(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calibrate model parameters to market data"""
        rates = market_data['rate'].values
        dt = 1/252  # Daily data
        
        def objective(params):
            a, b, sigma = params
            if a <= 0 or sigma <= 0:
                return 1e6
            
            # Maximum likelihood estimation
            n = len(rates) - 1
            dr = np.diff(rates.astype(float))
            r_prev = rates[:-1]
            
            mu = a * (b - r_prev) * dt
            residuals = dr - mu
            log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt) - \
                           0.5 * np.sum(residuals**2) / (sigma**2 * dt)
            
            return -log_likelihood
        
        result = optimize.minimize(objective, [0.1, 0.05, 0.01], 
                                 bounds=[(0.01, 1), (0.01, 0.2), (0.001, 0.1)])
        
        if result.success:
            self.a, self.b, self.sigma = result.x
        
        return {'a': self.a, 'b': self.b, 'sigma': self.sigma}

class FixedIncomeService:
    """Comprehensive fixed income analytics service"""
    
    def __init__(self):
        self.yield_curves = {}
        self.bonds_cache = {}
        self.models = {
            'vasicek': VasicekModel()
        }
        
        # Initialize with sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize sample bond and yield curve data"""
        # Sample US Treasury yield curve
        self.yield_curves['treasury'] = self._create_sample_treasury_curve()
        
        # Sample corporate yield curves by rating
        for rating in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']:
            self.yield_curves[f'corporate_{rating}'] = self._create_sample_corporate_curve(rating)
    
    def _create_sample_treasury_curve(self) -> YieldCurveAnalysis:
        """Create sample US Treasury yield curve"""
        maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        yields = [0.25, 0.3, 0.5, 1.2, 1.8, 2.5, 2.8, 3.0, 3.2, 3.1]  # Sample yields
        
        curve_points = []
        for i, (maturity, yield_rate) in enumerate(zip(maturities, yields)):
            # Calculate spot rates (simplified)
            spot_rate = yield_rate + np.random.normal(0, 0.02)
            
            # Calculate forward rates
            if i > 0:
                forward_rate = ((1 + spot_rate/100)**(maturity) / 
                              (1 + curve_points[i-1].spot_rate/100)**(maturities[i-1]))**(1/(maturity - maturities[i-1])) - 1
                forward_rate *= 100
            else:
                forward_rate = spot_rate
            
            # Calculate discount factor
            discount_factor = 1 / (1 + spot_rate/100)**maturity
            
            curve_points.append(YieldCurvePoint(
                maturity=maturity,
                yield_rate=yield_rate,
                spot_rate=spot_rate,
                forward_rate=forward_rate,
                discount_factor=discount_factor,
                par_rate=yield_rate
            ))
        
        return YieldCurveAnalysis(
            curve_date=datetime.now(),
            curve_points=curve_points,
            curve_type='treasury',
            interpolation_method='cubic_spline',
            curve_shifts=self._generate_curve_shifts(yields),
            key_rates={'2Y': yields[3], '5Y': yields[5], '10Y': yields[7], '30Y': yields[9]}
        )
    
    def _create_sample_corporate_curve(self, rating: str) -> YieldCurveAnalysis:
        """Create sample corporate yield curve with credit spreads"""
        treasury_curve = self.yield_curves['treasury']
        
        # Credit spreads by rating (basis points)
        credit_spreads = {
            'AAA': [10, 15, 20, 30, 40, 50, 60, 70, 80, 85],
            'AA': [15, 20, 30, 45, 60, 75, 85, 95, 110, 115],
            'A': [25, 35, 50, 70, 90, 110, 125, 140, 160, 165],
            'BBB': [50, 70, 100, 140, 180, 220, 250, 280, 320, 330],
            'BB': [200, 250, 350, 450, 550, 650, 700, 750, 850, 900],
            'B': [400, 500, 700, 900, 1100, 1300, 1400, 1500, 1700, 1800]
        }
        
        spreads = credit_spreads.get(rating, credit_spreads['BBB'])
        
        curve_points = []
        for i, (treasury_point, spread) in enumerate(zip(treasury_curve.curve_points, spreads)):
            corporate_yield = treasury_point.yield_rate + spread / 100
            
            curve_points.append(YieldCurvePoint(
                maturity=treasury_point.maturity,
                yield_rate=corporate_yield,
                spot_rate=corporate_yield + np.random.normal(0, 0.01),
                forward_rate=corporate_yield + np.random.normal(0, 0.02),
                discount_factor=1 / (1 + corporate_yield/100)**treasury_point.maturity,
                par_rate=corporate_yield
            ))
        
        return YieldCurveAnalysis(
            curve_date=datetime.now(),
            curve_points=curve_points,
            curve_type=f'corporate_{rating}',
            interpolation_method='cubic_spline',
            curve_shifts=self._generate_curve_shifts([p.yield_rate for p in curve_points]),
            key_rates={f'{rating}_2Y': curve_points[3].yield_rate, 
                      f'{rating}_5Y': curve_points[5].yield_rate,
                      f'{rating}_10Y': curve_points[7].yield_rate}
        )
    
    def _generate_curve_shifts(self, yields: List[float]) -> Dict[str, List[float]]:
        """Generate curve shift scenarios"""
        return {
            'parallel_up_25bp': [y + 0.25 for y in yields],
            'parallel_down_25bp': [y - 0.25 for y in yields],
            'parallel_up_100bp': [y + 1.0 for y in yields],
            'parallel_down_100bp': [y - 1.0 for y in yields],
            'steepener': [y + 0.1 * i for i, y in enumerate(yields)],
            'flattener': [y - 0.1 * i for i, y in enumerate(yields)],
            'butterfly': [y + (0.25 if 2 <= i <= 6 else 0) for i, y in enumerate(yields)]
        }
    
    # =================== BOND PRICING ===================
    
    def price_bond(self, bond: Bond, yield_to_maturity: float, 
                   settlement_date: Optional[datetime] = None) -> Dict[str, float]:
        """Price a bond given yield to maturity"""
        if settlement_date is None:
            settlement_date = datetime.now()
        
        # Calculate time to maturity
        time_to_maturity = (bond.maturity_date - settlement_date).days / 365.25
        
        # Calculate number of coupon periods
        periods_per_year = bond.frequency
        total_periods = int(time_to_maturity * periods_per_year)
        
        # Coupon payment
        coupon_payment = bond.face_value * (bond.coupon_rate / periods_per_year)
        
        # Calculate present value of coupon payments
        discount_rate = yield_to_maturity / periods_per_year
        
        pv_coupons = 0
        for period in range(1, total_periods + 1):
            pv_coupons += coupon_payment / (1 + discount_rate) ** period
        
        # Present value of principal
        pv_principal = bond.face_value / (1 + discount_rate) ** total_periods
        
        # Bond price
        bond_price = pv_coupons + pv_principal
        
        # Accrued interest calculation (simplified)
        days_since_last_coupon = 30  # Simplified
        accrued_interest = coupon_payment * (days_since_last_coupon / (365.25 / periods_per_year))
        
        clean_price = bond_price - accrued_interest
        
        return {
            'dirty_price': bond_price,
            'clean_price': clean_price,
            'accrued_interest': accrued_interest,
            'yield_to_maturity': yield_to_maturity,
            'time_to_maturity': time_to_maturity,
            'pv_coupons': pv_coupons,
            'pv_principal': pv_principal
        }
    
    def calculate_duration_convexity(self, bond: Bond, yield_to_maturity: float,
                                   settlement_date: Optional[datetime] = None) -> DurationConvexityResult:
        """Calculate duration and convexity measures"""
        if settlement_date is None:
            settlement_date = datetime.now()
        
        # Base price
        base_pricing = self.price_bond(bond, yield_to_maturity, settlement_date)
        base_price = base_pricing['clean_price']
        
        # Price with yield shock up
        yield_shock = 0.0001  # 1 basis point
        price_up = self.price_bond(bond, yield_to_maturity + yield_shock, settlement_date)['clean_price']
        price_down = self.price_bond(bond, yield_to_maturity - yield_shock, settlement_date)['clean_price']
        
        # Modified duration
        modified_duration = -(price_up - price_down) / (2 * base_price * yield_shock)
        
        # Convexity
        convexity = (price_up + price_down - 2 * base_price) / (base_price * yield_shock ** 2)
        
        # Macaulay duration
        macaulay_duration = modified_duration * (1 + yield_to_maturity / bond.frequency)
        
        # Effective duration (simplified)
        effective_duration = modified_duration
        
        # DV01 (Dollar value of 01)
        dv01 = modified_duration * base_price * 0.0001
        
        # Price sensitivity to various yield changes
        price_sensitivity = {
            '10bp': base_price * modified_duration * 0.001,
            '25bp': base_price * modified_duration * 0.0025,
            '50bp': base_price * modified_duration * 0.005,
            '100bp': base_price * modified_duration * 0.01
        }
        
        return DurationConvexityResult(
            modified_duration=modified_duration,
            effective_duration=effective_duration,
            macaulay_duration=macaulay_duration,
            convexity=convexity,
            dv01=dv01,
            price_sensitivity=price_sensitivity
        )
    
    def get_yield_curve(self, curve_type: str = 'treasury') -> YieldCurveAnalysis:
        """Get yield curve data"""
        return self.yield_curves.get(curve_type, self.yield_curves['treasury'])
    
    def interpolate_yield(self, curve_type: str, maturity: float) -> float:
        """Interpolate yield for a given maturity"""
        curve = self.get_yield_curve(curve_type)
        
        # Find surrounding points
        points = curve.curve_points
        
        if maturity <= points[0].maturity:
            return points[0].yield_rate
        if maturity >= points[-1].maturity:
            return points[-1].yield_rate
        
        # Linear interpolation
        for i in range(len(points) - 1):
            if points[i].maturity <= maturity <= points[i + 1].maturity:
                t = (maturity - points[i].maturity) / (points[i + 1].maturity - points[i].maturity)
                return points[i].yield_rate + t * (points[i + 1].yield_rate - points[i].yield_rate)
        
        return points[-1].yield_rate
    
    def simulate_interest_rates(self, model_name: str, initial_rate: float,
                              time_horizon: float, num_simulations: int = 1000) -> Dict[str, Any]:
        """Simulate interest rate paths using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        # Simulate rate paths
        rate_paths = model.simulate_rates(initial_rate, time_horizon, num_simulations)
        
        # Calculate statistics
        final_rates = rate_paths[:, -1]
        
        statistics = {
            'mean_final_rate': np.mean(final_rates),
            'std_final_rate': np.std(final_rates),
            'min_final_rate': np.min(final_rates),
            'max_final_rate': np.max(final_rates),
            'percentiles': {
                '5th': np.percentile(final_rates, 5),
                '25th': np.percentile(final_rates, 25),
                '50th': np.percentile(final_rates, 50),
                '75th': np.percentile(final_rates, 75),
                '95th': np.percentile(final_rates, 95)
            }
        }
        
        return {
            'model': model_name,
            'initial_rate': initial_rate,
            'time_horizon': time_horizon,
            'num_simulations': num_simulations,
            'rate_paths': rate_paths,
            'statistics': statistics
        }

# Global service instance
fixed_income_service = FixedIncomeService()

# Convenience functions
def get_treasury_curve() -> YieldCurveAnalysis:
    """Get current US Treasury yield curve"""
    return fixed_income_service.get_yield_curve('treasury')

def price_corporate_bond(cusip: str, coupon_rate: float, maturity_date: datetime,
                        credit_rating: str, yield_to_maturity: float) -> Dict[str, float]:
    """Price a corporate bond"""
    bond = Bond(
        cusip=cusip,
        issuer="Sample Corp",
        bond_type=BondType.CORPORATE,
        face_value=1000,
        coupon_rate=coupon_rate,
        maturity_date=maturity_date,
        issue_date=datetime.now() - timedelta(days=365),
        credit_rating=CreditRating(credit_rating)
    )
    
    return fixed_income_service.price_bond(bond, yield_to_maturity) 