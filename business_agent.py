"""
Business Analytics Agent - Production Grade
Version: 3.0.0
Features:
- Complete business metrics analysis
- Automated JSON reporting
- Comprehensive testing suite
- Advanced logging
- Error handling
"""

from langgraph.graph import StateGraph
from typing import TypedDict, Dict, List, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass
import sys
import hashlib

# Constants
MIN_REVENUE = 0.0
MIN_CUSTOMERS = 1
CAC_INCREASE_THRESHOLD = 20.0
REVENUE_GROWTH_THRESHOLD = 10.0
PROFIT_WARNING_THRESHOLD = 0.0
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
REPORT_VERSION = "1.0"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('business_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BusinessConfig:
    """Configuration for business analytics"""
    output_dir: str = "reports"
    decimal_precision: int = 2
    enable_backup: bool = True
    max_reports: int = 30

class BusinessState(TypedDict):
    """Data structure for analysis workflow"""
    daily_revenue: float
    daily_cost: float
    num_customers: int
    prev_day_revenue: float
    prev_day_cost: float
    prev_day_customers: int
    profit: Optional[float]
    revenue_change_pct: Optional[float]
    cost_change_pct: Optional[float]
    current_cac: Optional[float]
    prev_cac: Optional[float]
    cac_change_pct: Optional[float]
    alerts: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

class BusinessAnalyticsAgent:
    """Production-grade business analytics solution"""
    
    VERSION = "3.0.0"
    
    def __init__(self, config: BusinessConfig = BusinessConfig()):
        self.config = config
        self.workflow = StateGraph(BusinessState)
        self._setup_environment()
        self._build_graph()
        self.agent = self.workflow.compile()
        logger.info(f"BusinessAnalyticsAgent v{self.VERSION} initialized")

    def _setup_environment(self):
        """Initialize directories and environment"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if self.config.enable_backup:
            self.backup_dir = self.output_dir / "backups"
            self.backup_dir.mkdir(exist_ok=True)
        
        self._cleanup_old_reports()

    def _cleanup_old_reports(self):
        """Maintain report directory size"""
        reports = sorted(self.output_dir.glob("business_report_*.json"))
        if len(reports) > self.config.max_reports:
            for old_report in reports[:-self.config.max_reports]:
                old_report.unlink()
                logger.debug(f"Removed old report: {old_report.name}")

    def _build_graph(self):
        """Construct the analysis workflow"""
        self.workflow.add_node("validate_input", self._validate_input_node)
        self.workflow.add_node("calculate_metrics", self._metrics_node)
        self.workflow.add_node("generate_insights", self._insights_node)
        self.workflow.add_node("prepare_report", self._report_node)
        
        self.workflow.add_edge("validate_input", "calculate_metrics")
        self.workflow.add_edge("calculate_metrics", "generate_insights")
        self.workflow.add_edge("generate_insights", "prepare_report")
        
        self.workflow.set_entry_point("validate_input")
        self.workflow.set_finish_point("prepare_report")

    def _validate_input_node(self, state: BusinessState) -> BusinessState:
        """Validate and sanitize input data"""
        required_fields = {
            'daily_revenue': (float, MIN_REVENUE),
            'daily_cost': (float, MIN_REVENUE),
            'num_customers': (int, MIN_CUSTOMERS),
            'prev_day_revenue': (float, MIN_REVENUE),
            'prev_day_cost': (float, MIN_REVENUE)
        }
        
        for field, (field_type, min_value) in required_fields.items():
            if field not in state:
                raise ValueError(f"Missing required field: {field}")
            
            try:
                state[field] = field_type(state[field])
                if state[field] < min_value:
                    logger.warning(f"Low value warning for {field}: {state[field]}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid value for {field}: {e}")

        state.setdefault('prev_day_customers', state['num_customers'])
        state['alerts'] = []
        state['recommendations'] = []
        state['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'agent_version': self.VERSION,
            'report_version': REPORT_VERSION,
            'data_hash': self._generate_data_hash(state)
        }
        
        return state

    def _generate_data_hash(self, state: BusinessState) -> str:
        """Generate hash for input data validation"""
        hash_data = {
            'revenue': state['daily_revenue'],
            'cost': state['daily_cost'],
            'customers': state['num_customers'],
            'timestamp': datetime.now().timestamp()
        }
        return hashlib.md5(json.dumps(hash_data).encode()).hexdigest()

    def _metrics_node(self, state: BusinessState) -> BusinessState:
        """Calculate all business metrics"""
        try:
            # Core financial metrics
            state['profit'] = self._calculate_profit(
                state['daily_revenue'],
                state['daily_cost']
            )
            
            # Trend analysis
            state['revenue_change_pct'] = self._calculate_change_percentage(
                current=state['daily_revenue'],
                previous=state['prev_day_revenue'],
                metric_name="revenue"
            )
            
            state['cost_change_pct'] = self._calculate_change_percentage(
                current=state['daily_cost'],
                previous=state['prev_day_cost'],
                metric_name="cost"
            )
            
            # Customer metrics
            state['current_cac'] = self._calculate_cac(
                cost=state['daily_cost'],
                customers=state['num_customers']
            )
            
            state['prev_cac'] = self._calculate_cac(
                cost=state['prev_day_cost'],
                customers=state['prev_day_customers']
            )
            
            state['cac_change_pct'] = self._calculate_change_percentage(
                current=state['current_cac'],
                previous=state['prev_cac'],
                metric_name="CAC"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise

    def _insights_node(self, state: BusinessState) -> BusinessState:
        """Generate actionable business insights"""
        # Profit analysis
        if state['profit'] < PROFIT_WARNING_THRESHOLD:
            alert_msg = "ALERT: Negative profit detected"
            recommendation = "Implement cost reduction measures and explore revenue growth opportunities"
            state['alerts'].append(alert_msg)
            state['recommendations'].append(recommendation)
            logger.warning(alert_msg)
        
        # Revenue growth analysis
        if state['revenue_change_pct'] > REVENUE_GROWTH_THRESHOLD:
            recommendation = "Allocate additional budget to high-performing marketing channels"
            state['recommendations'].append(recommendation)
            logger.info("Significant revenue growth detected")
        
        # CAC analysis
        if state['cac_change_pct'] > CAC_INCREASE_THRESHOLD:
            rounded_change = round(state['cac_change_pct'], self.config.decimal_precision)
            alert_msg = f"ALERT: CAC increased by {rounded_change}% (Threshold: {CAC_INCREASE_THRESHOLD}%)"
            recommendation = "Audit marketing campaigns and optimize acquisition channels"
            state['alerts'].append(alert_msg)
            state['recommendations'].append(recommendation)
            logger.warning(alert_msg)
        
        return state

    def _report_node(self, state: BusinessState) -> Dict[str, Any]:
        """Generate comprehensive business report"""
        try:
            report = {
                'metadata': state['metadata'],
                'financial_metrics': {
                    'revenue': {
                        'current': round(state['daily_revenue'], self.config.decimal_precision),
                        'previous': round(state['prev_day_revenue'], self.config.decimal_precision),
                        'change_pct': round(state['revenue_change_pct'], self.config.decimal_precision),
                        'trend': 'up' if state['revenue_change_pct'] > 0 else 'down'
                    },
                    'cost': {
                        'current': round(state['daily_cost'], self.config.decimal_precision),
                        'previous': round(state['prev_day_cost'], self.config.decimal_precision),
                        'change_pct': round(state['cost_change_pct'], self.config.decimal_precision),
                        'trend': 'up' if state['cost_change_pct'] > 0 else 'down'
                    },
                    'profit': {
                        'amount': round(state['profit'], self.config.decimal_precision),
                        'margin': self._calculate_profit_margin(
                            state['daily_revenue'],
                            state['profit']
                        ),
                        'status': 'positive' if state['profit'] >= 0 else 'negative'
                    }
                },
                'customer_metrics': {
                    'count': state['num_customers'],
                    'cac': {
                        'current': round(state['current_cac'], self.config.decimal_precision),
                        'previous': round(state['prev_cac'], self.config.decimal_precision),
                        'change_pct': round(state['cac_change_pct'], self.config.decimal_precision),
                        'status': 'critical' if state['cac_change_pct'] > CAC_INCREASE_THRESHOLD else 'normal'
                    }
                },
                'insights': {
                    'alerts': state['alerts'],
                    'recommendations': state['recommendations'],
                    'priority': 'high' if len(state['alerts']) > 0 else 'normal'
                },
                'system': {
                    'timestamp': datetime.now().strftime(DATE_FORMAT),
                    'processing_time_ms': int((datetime.now().timestamp() - 
                                             datetime.fromisoformat(state['metadata']['analysis_date']).timestamp()) * 1000)
                }
            }
            
            # Save reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"business_report_{timestamp}.json"
            self._save_report(report, report_filename)
            
            if self.config.enable_backup:
                backup_filename = f"backup_{report_filename}"
                self._save_report(report, backup_filename, backup=True)
            
            logger.info(f"Report generated: {report_filename}")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    def _save_report(self, data: Dict, filename: str, backup: bool = False):
        """Save report to JSON file with validation"""
        try:
            target_dir = self.backup_dir if backup else self.output_dir
            filepath = target_dir / filename
            
            # Validate data before saving
            self._validate_report_data(data)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            raise

    def _validate_report_data(self, data: Dict):
        """Validate report data structure"""
        required_sections = ['metadata', 'financial_metrics', 'customer_metrics', 'insights']
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Invalid report format: missing {section} section")

    @staticmethod
    def _calculate_profit(revenue: float, cost: float) -> float:
        """Calculate daily profit (revenue - cost)"""
        return revenue - cost

    @staticmethod
    def _calculate_profit_margin(revenue: float, profit: float) -> float:
        """Calculate profit margin percentage"""
        if revenue == 0:
            return 0.0
        return (profit / revenue) * 100

    @staticmethod
    def _calculate_change_percentage(current: float, previous: float, metric_name: str = "") -> float:
        """Calculate percentage change with validation"""
        if previous == 0:
            logger.warning(f"Cannot calculate percentage change for {metric_name} - previous value is zero")
            return float('inf') if current > 0 else float('-inf')
        return ((current - previous) / previous) * 100

    @staticmethod
    def _calculate_cac(cost: float, customers: int) -> float:
        """Calculate Customer Acquisition Cost"""
        if customers < MIN_CUSTOMERS:
            logger.warning(f"Customer count below minimum ({MIN_CUSTOMERS})")
            return float('inf')
        return cost / customers

    def analyze(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full analysis workflow"""
        try:
            logger.info("Starting business analysis")
            result = self.agent.invoke(business_data)
            logger.info("Analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    def test_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive test cases"""
        test_cases = [
            {
                "name": "Growing business scenario",
                "input": {
                    "daily_revenue": 15000,
                    "daily_cost": 8000,
                    "num_customers": 100,
                    "prev_day_revenue": 12000,
                    "prev_day_cost": 7000,
                    "prev_day_customers": 90
                },
                "expected_alerts": [],
                "expected_recommendations": [
                    "Allocate additional budget to high-performing marketing channels"
                ]
            },
            {
                "name": "Declining business scenario",
                "input": {
                    "daily_revenue": 8000,
                    "daily_cost": 9000,
                    "num_customers": 80,
                    "prev_day_revenue": 12000,
                    "prev_day_cost": 7000,
                    "prev_day_customers": 90
                },
                "expected_alerts": [
                    "ALERT: Negative profit detected",
                    f"ALERT: CAC increased by 44.64% (Threshold: {CAC_INCREASE_THRESHOLD}%)"
                ],
                "expected_recommendations": [
                    "Implement cost reduction measures and explore revenue growth opportunities",
                    "Audit marketing campaigns and optimize acquisition channels"
                ]
            }
        ]
        
        test_results = []
        for case in test_cases:
            try:
                logger.info(f"Running test: {case['name']}")
                result = self.analyze(case["input"])
                
                # Verify alerts
                assert set(case["expected_alerts"]) == set(result["insights"]["alerts"]), (
                    f"Alerts mismatch in {case['name']}\n"
                    f"Expected: {case['expected_alerts']}\n"
                    f"Got: {result['insights']['alerts']}"
                )
                
                # Verify recommendations
                assert set(case["expected_recommendations"]) == set(result["insights"]["recommendations"]), (
                    f"Recommendations mismatch in {case['name']}\n"
                    f"Expected: {case['expected_recommendations']}\n"
                    f"Got: {result['insights']['recommendations']}"
                )
                
                test_results.append({
                    "test_case": case["name"],
                    "status": "passed",
                    "timestamp": datetime.now().strftime(DATE_FORMAT)
                })
                logger.info(f"Test passed: {case['name']}")
                
            except AssertionError as e:
                logger.error(f"Test failed: {case['name']}\n{str(e)}")
                test_results.append({
                    "test_case": case["name"],
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().strftime(DATE_FORMAT)
                })
        
        # Generate test report
        test_report = {
            "summary": {
                "total_tests": len(test_cases),
                "passed": sum(1 for r in test_results if r["status"] == "passed"),
                "failed": sum(1 for r in test_results if r["status"] == "failed"),
                "success_rate": f"{sum(1 for r in test_results if r['status'] == 'passed') / len(test_cases) * 100:.1f}%"
            },
            "details": test_results
        }
        
        # Save test report
        test_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_report(test_report, test_filename)
        
        return test_report

def main():
    """Entry point for the application"""
    try:
        # Configuration
        config = BusinessConfig(
            output_dir="business_reports",
            decimal_precision=2,
            enable_backup=True,
            max_reports=30
        )
        
        # Initialize agent
        agent = BusinessAnalyticsAgent(config)
        logger.info("Business Analytics Agent initialized successfully")
        
        # Sample data analysis
        sample_data = {
            "daily_revenue": 12000,
            "daily_cost": 8500,
            "num_customers": 95,
            "prev_day_revenue": 10000,
            "prev_day_cost": 8000,
            "prev_day_customers": 90
        }
        
        print("Running business analysis...")
        report = agent.analyze(sample_data)
        print("\nAnalysis Report Summary:")
        print(json.dumps({
            "profit": report["financial_metrics"]["profit"],
            "alerts": report["insights"]["alerts"],
            "recommendations": report["insights"]["recommendations"]
        }, indent=2))
        
        # Run tests
        print("\nRunning test suite...")
        test_results = agent.test_analysis()
        print("\nTest Results Summary:")
        print(json.dumps(test_results["summary"], indent=2))
        
    except Exception as e:
        logger.critical(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()