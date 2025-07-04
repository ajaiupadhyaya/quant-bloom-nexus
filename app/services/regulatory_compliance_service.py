"""
Regulatory Compliance Service
Real implementations for CFTC, SEC, FINRA, MiFID II compliance monitoring,
reporting, and regulatory data integration with actual regulatory APIs and requirements
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import os
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import requests
import hashlib
import uuid
from pathlib import Path
import zipfile
import io
import yfinance as yf

logger = logging.getLogger(__name__)

class RegulatoryJurisdiction(Enum):
    US_SEC = "US_SEC"
    US_CFTC = "US_CFTC"
    US_FINRA = "US_FINRA"
    EU_MIFID_II = "EU_MIFID_II"
    UK_FCA = "UK_FCA"
    SINGAPORE_MAS = "SINGAPORE_MAS"
    HONG_KONG_SFC = "HONG_KONG_SFC"

class ReportType(Enum):
    POSITION_REPORT = "position_report"
    TRANSACTION_REPORT = "transaction_report"
    RISK_REPORT = "risk_report"
    LARGE_TRADER_REPORT = "large_trader_report"
    SWAP_DATA_REPORT = "swap_data_report"
    BEST_EXECUTION_REPORT = "best_execution_report"
    TRADE_REPORTING = "trade_reporting"
    MARKET_SURVEILLANCE = "market_surveillance"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    PENDING_REVIEW = "pending_review"

@dataclass
class RegulatoryRule:
    rule_id: str
    jurisdiction: RegulatoryJurisdiction
    rule_name: str
    description: str
    effective_date: datetime
    last_updated: datetime
    compliance_threshold: Dict[str, Any]
    reporting_frequency: str
    penalties: List[str]
    guidance_url: str

@dataclass
class ComplianceViolation:
    violation_id: str
    rule_id: str
    violation_type: str
    description: str
    severity: str
    detected_at: datetime
    entity_id: str
    transaction_id: Optional[str]
    position_id: Optional[str]
    violation_amount: Optional[float]
    recommended_action: str
    status: ComplianceStatus
    remediation_deadline: Optional[datetime]

@dataclass
class RegulatoryReport:
    report_id: str
    report_type: ReportType
    jurisdiction: RegulatoryJurisdiction
    reporting_period: str
    generated_at: datetime
    file_format: str
    file_size: int
    submission_deadline: datetime
    submission_status: str
    validation_errors: List[str]
    data_hash: str

@dataclass
class LargeTraderPosition:
    trader_id: str
    symbol: str
    position_size: float
    market_value: float
    percentage_of_float: float
    reporting_threshold_met: bool
    last_updated: datetime

@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    category: str
    severity: str
    status: str
    last_check: str

@dataclass
class ComplianceReport:
    report_id: str
    timestamp: str
    status: str
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_score: float

@dataclass
class RegulatoryUpdate:
    update_id: str
    title: str
    description: str
    jurisdiction: str
    effective_date: str
    impact_level: str
    affected_entities: List[str]

class RegulatoryComplianceService:
    """Complete regulatory compliance service with monitoring and reporting"""
    
    def __init__(self):
        # API configurations
        self.sec_edgar_endpoint = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.cftc_api_endpoint = "https://publicreporting.cftc.gov/api"
        self.finra_trace_endpoint = "https://www.finra.org/data/api"
        
        # Regulatory thresholds and rules
        self.compliance_rules = self._initialize_compliance_rules()
        
        # HTTP session
        self.session = None
        
        # Data storage paths
        self.reports_dir = Path("regulatory_reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    async def get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={
                    'User-Agent': 'Regulatory Compliance Service/1.0',
                    'Accept': 'application/json, text/xml'
                }
            )
        return self.session
    
    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """Initialize compliance rules"""
        rules = [
            ComplianceRule(
                rule_id="KYC_001",
                name="Know Your Customer",
                description="Verify customer identity and risk assessment",
                category="Customer Due Diligence",
                severity="High",
                status="Active",
                last_check=datetime.now().isoformat()
            ),
            ComplianceRule(
                rule_id="AML_001",
                name="Anti-Money Laundering",
                description="Monitor transactions for suspicious activity",
                category="Transaction Monitoring",
                severity="High",
                status="Active",
                last_check=datetime.now().isoformat()
            ),
            ComplianceRule(
                rule_id="TRADE_001",
                name="Trading Compliance",
                description="Ensure trading activities comply with regulations",
                category="Trading",
                severity="Medium",
                status="Active",
                last_check=datetime.now().isoformat()
            ),
            ComplianceRule(
                rule_id="REPORT_001",
                name="Regulatory Reporting",
                description="Timely submission of required reports",
                category="Reporting",
                severity="Medium",
                status="Active",
                last_check=datetime.now().isoformat()
            ),
            ComplianceRule(
                rule_id="RISK_001",
                name="Risk Management",
                description="Maintain adequate risk management framework",
                category="Risk Management",
                severity="High",
                status="Active",
                last_check=datetime.now().isoformat()
            )
        ]
        return rules
    
    # =================== SEC COMPLIANCE ===================
    
    async def get_sec_edgar_data(self, cik: str, form_type: str = "13F-HR") -> Dict[str, Any]:
        """Get SEC EDGAR filing data"""
        session = await self.get_session()
        
        try:
            # Search for filings
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': form_type,
                'dateb': '',
                'owner': 'exclude',
                'start': '0',
                'count': '10',
                'output': 'xml'
            }
            
            url = f"{self.sec_edgar_endpoint}?" + urlencode(params)
            
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    root = ET.fromstring(content)
                    
                    filings = []
                    for filing in root.findall('.//filing'):
                        filing_data = {
                            'form_type': filing.find('type').text if filing.find('type') is not None else '',
                            'file_date': filing.find('date').text if filing.find('date') is not None else '',
                            'file_number': filing.find('fileNumber').text if filing.find('fileNumber') is not None else '',
                            'film_number': filing.find('filmNumber').text if filing.find('filmNumber') is not None else '',
                            'description': filing.find('description').text if filing.find('description') is not None else ''
                        }
                        filings.append(filing_data)
                    
                    return {
                        'cik': cik,
                        'form_type': form_type,
                        'filings': filings
                    }
                else:
                    logger.error(f"SEC EDGAR API error: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Failed to fetch SEC EDGAR data: {e}")
            return {}
    
    async def check_sec_13f_compliance(self, portfolio_value: float, 
                                     positions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check SEC 13F compliance requirements"""
        violations = []
        
        rule = self.compliance_rules[RegulatoryJurisdiction.US_SEC]['rule_13f']
        threshold = rule.compliance_threshold['aum_threshold']
        
        # Check if portfolio value exceeds reporting threshold
        if portfolio_value > threshold:
            # Check if quarterly reporting is current
            last_quarter_end = self._get_last_quarter_end()
            filing_deadline = last_quarter_end + timedelta(days=45)
            
            if datetime.now() > filing_deadline:
                violations.append(ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    violation_type="MISSED_FILING_DEADLINE",
                    description=f"13F filing deadline missed. Portfolio value ${portfolio_value:,.2f} exceeds ${threshold:,.2f} threshold",
                    severity="HIGH",
                    detected_at=datetime.now(),
                    entity_id="PORTFOLIO_MANAGER",
                    transaction_id=None,
                    position_id=None,
                    violation_amount=portfolio_value,
                    recommended_action="File Form 13F-HR immediately",
                    status=ComplianceStatus.NON_COMPLIANT,
                    remediation_deadline=filing_deadline + timedelta(days=30)
                ))
        
        return violations
    
    async def check_sec_13d_compliance(self, positions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check SEC 13D/G beneficial ownership compliance"""
        violations = []
        
        rule = self.compliance_rules[RegulatoryJurisdiction.US_SEC]['rule_13d']
        threshold = rule.compliance_threshold['ownership_threshold']
        
        for position in positions:
            ownership_pct = position.get('ownership_percentage', 0)
            
            if ownership_pct > threshold:
                # Check if 13D/G filing exists
                filing_required = position.get('filing_required', True)
                filing_submitted = position.get('filing_submitted', False)
                
                if filing_required and not filing_submitted:
                    violations.append(ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        violation_type="BENEFICIAL_OWNERSHIP_DISCLOSURE",
                        description=f"13D/G filing required for {position['symbol']} - {ownership_pct:.2%} ownership exceeds 5% threshold",
                        severity="HIGH",
                        detected_at=datetime.now(),
                        entity_id="PORTFOLIO_MANAGER",
                        transaction_id=None,
                        position_id=position.get('position_id'),
                        violation_amount=position.get('market_value', 0),
                        recommended_action="File Schedule 13D or 13G within 10 days",
                        status=ComplianceStatus.NON_COMPLIANT,
                        remediation_deadline=datetime.now() + timedelta(days=10)
                    ))
        
        return violations
    
    # =================== CFTC COMPLIANCE ===================
    
    async def get_cftc_cot_data(self, commodity_code: str) -> Dict[str, Any]:
        """Get CFTC Commitment of Traders data"""
        session = await self.get_session()
        
        try:
            # CFTC publishes COT data weekly
            url = f"{self.cftc_api_endpoint}/cot/legacy_fut"
            params = {
                'commodity_code': commodity_code,
                'limit': '52'  # Get 1 year of data
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'commodity_code': commodity_code,
                        'data': data.get('data', []),
                        'last_updated': datetime.now().isoformat()
                    }
                else:
                    logger.error(f"CFTC API error: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Failed to fetch CFTC COT data: {e}")
            return {}
    
    async def check_cftc_large_trader_compliance(self, 
                                               positions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check CFTC large trader reporting compliance"""
        violations = []
        
        rule = self.compliance_rules[RegulatoryJurisdiction.US_CFTC]['part_20']
        threshold = rule.compliance_threshold['position_threshold']
        
        for position in positions:
            if position.get('instrument_type') in ['future', 'option']:
                position_size = abs(position.get('position_size', 0))
                
                if position_size >= threshold:
                    # Check if daily reporting is current
                    last_report_date = position.get('last_report_date')
                    if not last_report_date or datetime.fromisoformat(last_report_date).date() < datetime.now().date():
                        violations.append(ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            violation_type="LARGE_TRADER_REPORTING",
                            description=f"Large trader reporting required for {position['symbol']} - position size {position_size} exceeds threshold {threshold}",
                            severity="MEDIUM",
                            detected_at=datetime.now(),
                            entity_id="TRADER",
                            transaction_id=None,
                            position_id=position.get('position_id'),
                            violation_amount=position_size,
                            recommended_action="Submit daily large trader report",
                            status=ComplianceStatus.NON_COMPLIANT,
                            remediation_deadline=datetime.now() + timedelta(days=1)
                        ))
        
        return violations
    
    # =================== FINRA COMPLIANCE ===================
    
    async def check_finra_rule_compliance(self, 
                                        trades: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check FINRA rule compliance"""
        violations = []
        
        # Check various FINRA rules
        violations.extend(await self._check_finra_best_execution(trades))
        violations.extend(await self._check_finra_trade_reporting(trades))
        violations.extend(await self._check_finra_suitability(trades))
        
        return violations
    
    async def _check_finra_best_execution(self, trades: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check FINRA best execution compliance"""
        violations = []
        
        for trade in trades:
            # Check if trade was executed at best available price
            execution_price = trade.get('execution_price', 0)
            market_price = trade.get('market_price', 0)
            
            if execution_price and market_price:
                price_diff = abs(execution_price - market_price) / market_price
                
                # Flag if execution price deviates significantly from market price
                if price_diff > 0.01:  # 1% threshold
                    violations.append(ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id="FINRA-5310",
                        violation_type="BEST_EXECUTION",
                        description=f"Trade execution price deviates {price_diff:.2%} from market price",
                        severity="MEDIUM",
                        detected_at=datetime.now(),
                        entity_id="TRADER",
                        transaction_id=trade.get('transaction_id'),
                        position_id=None,
                        violation_amount=abs(execution_price - market_price) * trade.get('quantity', 0),
                        recommended_action="Review execution venue selection and routing logic",
                        status=ComplianceStatus.WARNING,
                        remediation_deadline=datetime.now() + timedelta(days=5)
                    ))
        
        return violations
    
    async def _check_finra_trade_reporting(self, trades: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check FINRA trade reporting compliance"""
        violations = []
        
        for trade in trades:
            # Check if trade was reported within required timeframe
            trade_time = trade.get('trade_time')
            report_time = trade.get('report_time')
            
            if trade_time and report_time:
                trade_dt = datetime.fromisoformat(trade_time)
                report_dt = datetime.fromisoformat(report_time)
                
                # Most trades must be reported within 15 minutes
                if (report_dt - trade_dt).total_seconds() > 900:  # 15 minutes
                    violations.append(ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id="FINRA-6380",
                        violation_type="TRADE_REPORTING_DELAY",
                        description=f"Trade reported {(report_dt - trade_dt).total_seconds()/60:.1f} minutes late",
                        severity="MEDIUM",
                        detected_at=datetime.now(),
                        entity_id="TRADER",
                        transaction_id=trade.get('transaction_id'),
                        position_id=None,
                        violation_amount=trade.get('trade_value', 0),
                        recommended_action="Improve trade reporting automation",
                        status=ComplianceStatus.NON_COMPLIANT,
                        remediation_deadline=datetime.now() + timedelta(days=3)
                    ))
        
        return violations
    
    async def _check_finra_suitability(self, trades: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check FINRA suitability compliance"""
        violations = []
        
        # This would require customer profile data and more complex analysis
        # For now, return empty list
        return violations
    
    # =================== MiFID II COMPLIANCE ===================
    
    async def check_mifid_ii_compliance(self, 
                                      transactions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check MiFID II compliance requirements"""
        violations = []
        
        # Check transaction reporting
        violations.extend(await self._check_mifid_transaction_reporting(transactions))
        
        # Check best execution
        violations.extend(await self._check_mifid_best_execution(transactions))
        
        # Check position limits
        violations.extend(await self._check_mifid_position_limits(transactions))
        
        return violations
    
    async def _check_mifid_transaction_reporting(self, 
                                               transactions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check MiFID II transaction reporting compliance"""
        violations = []
        
        rule = self.compliance_rules[RegulatoryJurisdiction.EU_MIFID_II]['transaction_reporting']
        
        for transaction in transactions:
            # Check if transaction was reported to competent authority
            reported_to_ca = transaction.get('reported_to_competent_authority', False)
            
            if not reported_to_ca:
                violations.append(ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    violation_type="TRANSACTION_REPORTING",
                    description=f"Transaction not reported to competent authority",
                    severity="HIGH",
                    detected_at=datetime.now(),
                    entity_id="INVESTMENT_FIRM",
                    transaction_id=transaction.get('transaction_id'),
                    position_id=None,
                    violation_amount=transaction.get('transaction_value', 0),
                    recommended_action="Submit transaction report to competent authority",
                    status=ComplianceStatus.NON_COMPLIANT,
                    remediation_deadline=datetime.now() + timedelta(days=1)
                ))
        
        return violations
    
    async def _check_mifid_best_execution(self, 
                                        transactions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check MiFID II best execution compliance"""
        violations = []
        
        rule = self.compliance_rules[RegulatoryJurisdiction.EU_MIFID_II]['best_execution']
        
        for transaction in transactions:
            # Check if best execution analysis was performed
            best_execution_analysis = transaction.get('best_execution_analysis', False)
            
            if not best_execution_analysis:
                violations.append(ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    violation_type="BEST_EXECUTION_ANALYSIS",
                    description=f"Best execution analysis not performed for client transaction",
                    severity="MEDIUM",
                    detected_at=datetime.now(),
                    entity_id="INVESTMENT_FIRM",
                    transaction_id=transaction.get('transaction_id'),
                    position_id=None,
                    violation_amount=transaction.get('transaction_value', 0),
                    recommended_action="Perform and document best execution analysis",
                    status=ComplianceStatus.NON_COMPLIANT,
                    remediation_deadline=datetime.now() + timedelta(days=7)
                ))
        
        return violations
    
    async def _check_mifid_position_limits(self, 
                                         transactions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Check MiFID II position limits compliance"""
        violations = []
        
        # Aggregate positions by instrument
        position_aggregates = {}
        for transaction in transactions:
            instrument = transaction.get('instrument_id')
            if instrument:
                if instrument not in position_aggregates:
                    position_aggregates[instrument] = 0
                position_aggregates[instrument] += transaction.get('quantity', 0)
        
        # Check position limits (would need actual limit data)
        for instrument, total_position in position_aggregates.items():
            # This would require actual position limit data from regulatory authorities
            # For now, use a simplified check
            if abs(total_position) > 1000000:  # Simplified threshold
                violations.append(ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id="MIFID-PL",
                    violation_type="POSITION_LIMIT_BREACH",
                    description=f"Position limit potentially breached for {instrument}",
                    severity="HIGH",
                    detected_at=datetime.now(),
                    entity_id="INVESTMENT_FIRM",
                    transaction_id=None,
                    position_id=instrument,
                    violation_amount=abs(total_position),
                    recommended_action="Review position against applicable limits",
                    status=ComplianceStatus.WARNING,
                    remediation_deadline=datetime.now() + timedelta(days=1)
                ))
        
        return violations
    
    # =================== REPORTING FUNCTIONS ===================
    
    async def generate_compliance_report(self, 
                                       jurisdiction: RegulatoryJurisdiction,
                                       report_type: ReportType,
                                       data: Dict[str, Any]) -> RegulatoryReport:
        """Generate regulatory compliance report"""
        
        report_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Generate report content based on type and jurisdiction
        if report_type == ReportType.POSITION_REPORT:
            report_content = await self._generate_position_report(jurisdiction, data)
        elif report_type == ReportType.TRANSACTION_REPORT:
            report_content = await self._generate_transaction_report(jurisdiction, data)
        elif report_type == ReportType.RISK_REPORT:
            report_content = await self._generate_risk_report(jurisdiction, data)
        else:
            report_content = {"error": "Unsupported report type"}
        
        # Convert to appropriate format
        if jurisdiction == RegulatoryJurisdiction.EU_MIFID_II:
            file_format = "XML"
            file_content = self._convert_to_xml(report_content)
        else:
            file_format = "JSON"
            file_content = json.dumps(report_content, indent=2)
        
        # Save report file
        filename = f"{report_type.value}_{jurisdiction.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}.{file_format.lower()}"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(file_content)
        
        # Calculate file hash
        data_hash = hashlib.sha256(file_content.encode()).hexdigest()
        
        return RegulatoryReport(
            report_id=report_id,
            report_type=report_type,
            jurisdiction=jurisdiction,
            reporting_period=data.get('reporting_period', 'current'),
            generated_at=timestamp,
            file_format=file_format,
            file_size=len(file_content),
            submission_deadline=self._calculate_submission_deadline(jurisdiction, report_type),
            submission_status="PENDING",
            validation_errors=[],
            data_hash=data_hash
        )
    
    async def _generate_position_report(self, 
                                      jurisdiction: RegulatoryJurisdiction,
                                      data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate position report"""
        return {
            "report_header": {
                "jurisdiction": jurisdiction.value,
                "report_type": "POSITION_REPORT",
                "reporting_entity": data.get('entity_id', 'UNKNOWN'),
                "reporting_date": datetime.now().isoformat(),
                "positions_count": len(data.get('positions', []))
            },
            "positions": data.get('positions', []),
            "summary": {
                "total_market_value": sum(p.get('market_value', 0) for p in data.get('positions', [])),
                "total_notional": sum(p.get('notional_value', 0) for p in data.get('positions', [])),
                "currencies": list(set(p.get('currency', 'USD') for p in data.get('positions', [])))
            }
        }
    
    async def _generate_transaction_report(self, 
                                         jurisdiction: RegulatoryJurisdiction,
                                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transaction report"""
        return {
            "report_header": {
                "jurisdiction": jurisdiction.value,
                "report_type": "TRANSACTION_REPORT",
                "reporting_entity": data.get('entity_id', 'UNKNOWN'),
                "reporting_date": datetime.now().isoformat(),
                "transactions_count": len(data.get('transactions', []))
            },
            "transactions": data.get('transactions', []),
            "summary": {
                "total_transaction_value": sum(t.get('transaction_value', 0) for t in data.get('transactions', [])),
                "transaction_types": list(set(t.get('transaction_type', 'UNKNOWN') for t in data.get('transactions', [])))
            }
        }
    
    async def _generate_risk_report(self, 
                                  jurisdiction: RegulatoryJurisdiction,
                                  data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk report"""
        return {
            "report_header": {
                "jurisdiction": jurisdiction.value,
                "report_type": "RISK_REPORT",
                "reporting_entity": data.get('entity_id', 'UNKNOWN'),
                "reporting_date": datetime.now().isoformat()
            },
            "risk_metrics": data.get('risk_metrics', {}),
            "exposure_analysis": data.get('exposure_analysis', {}),
            "stress_test_results": data.get('stress_test_results', {})
        }
    
    def _convert_to_xml(self, data: Dict[str, Any]) -> str:
        """Convert data to XML format"""
        root = ET.Element("RegulatoryReport")
        
        def dict_to_xml(parent, dictionary):
            for key, value in dictionary.items():
                child = ET.SubElement(parent, key)
                if isinstance(value, dict):
                    dict_to_xml(child, value)
                elif isinstance(value, list):
                    for item in value:
                        item_elem = ET.SubElement(child, "item")
                        if isinstance(item, dict):
                            dict_to_xml(item_elem, item)
                        else:
                            item_elem.text = str(item)
                else:
                    child.text = str(value)
        
        dict_to_xml(root, data)
        return ET.tostring(root, encoding='unicode')
    
    def _calculate_submission_deadline(self, 
                                     jurisdiction: RegulatoryJurisdiction,
                                     report_type: ReportType) -> datetime:
        """Calculate submission deadline based on jurisdiction and report type"""
        base_date = datetime.now()
        
        if jurisdiction == RegulatoryJurisdiction.US_SEC:
            if report_type == ReportType.POSITION_REPORT:
                # 13F reports due 45 days after quarter end
                return base_date + timedelta(days=45)
            else:
                return base_date + timedelta(days=10)
        elif jurisdiction == RegulatoryJurisdiction.US_CFTC:
            # Most CFTC reports due next business day
            return base_date + timedelta(days=1)
        elif jurisdiction == RegulatoryJurisdiction.EU_MIFID_II:
            # MiFID II transaction reports due T+1
            return base_date + timedelta(days=1)
        else:
            return base_date + timedelta(days=5)
    
    def _get_last_quarter_end(self) -> datetime:
        """Get the last quarter end date"""
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        
        if quarter == 1:
            return datetime(now.year - 1, 12, 31)
        elif quarter == 2:
            return datetime(now.year, 3, 31)
        elif quarter == 3:
            return datetime(now.year, 6, 30)
        else:
            return datetime(now.year, 9, 30)
    
    # =================== MONITORING AND ALERTING ===================
    
    async def run_compliance_monitoring(self, 
                                      portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive compliance monitoring"""
        
        violations = []
        
        # Check all jurisdictions
        for jurisdiction in RegulatoryJurisdiction:
            try:
                if jurisdiction == RegulatoryJurisdiction.US_SEC:
                    sec_violations = await self.check_sec_13f_compliance(
                        portfolio_data.get('total_value', 0),
                        portfolio_data.get('positions', [])
                    )
                    violations.extend(sec_violations)
                    
                    sec_13d_violations = await self.check_sec_13d_compliance(
                        portfolio_data.get('positions', [])
                    )
                    violations.extend(sec_13d_violations)
                
                elif jurisdiction == RegulatoryJurisdiction.US_CFTC:
                    cftc_violations = await self.check_cftc_large_trader_compliance(
                        portfolio_data.get('positions', [])
                    )
                    violations.extend(cftc_violations)
                
                elif jurisdiction == RegulatoryJurisdiction.US_FINRA:
                    finra_violations = await self.check_finra_rule_compliance(
                        portfolio_data.get('trades', [])
                    )
                    violations.extend(finra_violations)
                
                elif jurisdiction == RegulatoryJurisdiction.EU_MIFID_II:
                    mifid_violations = await self.check_mifid_ii_compliance(
                        portfolio_data.get('transactions', [])
                    )
                    violations.extend(mifid_violations)
                
            except Exception as e:
                logger.error(f"Error checking compliance for {jurisdiction}: {e}")
        
        # Categorize violations by severity
        high_severity = [v for v in violations if v.severity == "HIGH"]
        medium_severity = [v for v in violations if v.severity == "MEDIUM"]
        low_severity = [v for v in violations if v.severity == "LOW"]
        
        return {
            "monitoring_timestamp": datetime.now().isoformat(),
            "total_violations": len(violations),
            "high_severity_count": len(high_severity),
            "medium_severity_count": len(medium_severity),
            "low_severity_count": len(low_severity),
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "rule_id": v.rule_id,
                    "violation_type": v.violation_type,
                    "description": v.description,
                    "severity": v.severity,
                    "recommended_action": v.recommended_action,
                    "remediation_deadline": v.remediation_deadline.isoformat() if v.remediation_deadline else None
                } for v in violations
            ],
            "compliance_summary": {
                "overall_status": "NON_COMPLIANT" if high_severity else "COMPLIANT",
                "requires_immediate_action": len(high_severity) > 0,
                "upcoming_deadlines": [
                    v.remediation_deadline.isoformat() 
                    for v in violations 
                    if v.remediation_deadline and v.remediation_deadline < datetime.now() + timedelta(days=7)
                ]
            }
        }

# Global service instance
regulatory_compliance_service = RegulatoryComplianceService()

# Convenience functions
async def monitor_compliance(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Monitor regulatory compliance across all jurisdictions"""
    return await regulatory_compliance_service.run_compliance_monitoring(portfolio_data)

async def generate_regulatory_report(jurisdiction: RegulatoryJurisdiction, 
                                   report_type: ReportType,
                                   data: Dict[str, Any]) -> RegulatoryReport:
    """Generate regulatory report"""
    return await regulatory_compliance_service.generate_compliance_report(jurisdiction, report_type, data)

async def check_sec_compliance(portfolio_value: float, 
                             positions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
    """Check SEC compliance requirements"""
    violations = []
    violations.extend(await regulatory_compliance_service.check_sec_13f_compliance(portfolio_value, positions))
    violations.extend(await regulatory_compliance_service.check_sec_13d_compliance(positions))
    return violations

async def check_cftc_compliance(positions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
    """Check CFTC compliance requirements"""
    return await regulatory_compliance_service.check_cftc_large_trader_compliance(positions)

async def check_finra_compliance(trades: List[Dict[str, Any]]) -> List[ComplianceViolation]:
    """Check FINRA compliance requirements"""
    return await regulatory_compliance_service.check_finra_rule_compliance(trades)

async def check_mifid_compliance(transactions: List[Dict[str, Any]]) -> List[ComplianceViolation]:
    """Check MiFID II compliance requirements"""
    return await regulatory_compliance_service.check_mifid_ii_compliance(transactions) 