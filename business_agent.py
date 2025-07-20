"""
Business Analytics Agent - Production Grade
Version: 3.1.0
Features:
- Complete business metrics analysis
- Automated JSON reporting
- Comprehensive testing suite
- Advanced logging
- Error handling
- Trend forecasting
- Financial ratios analysis
- HTML dashboard generation
- SQLite database integration
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
import sqlite3
from jinja2 import Template

# Constants
MIN_REVENUE = 0.0
MIN_CUSTOMERS = 1
CAC_INCREASE_THRESHOLD = 20.0
REVENUE_GROWTH_THRESHOLD = 10.0
PROFIT_WARNING_THRESHOLD = 0.0
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
REPORT_VERSION = "1.1"
DB_NAME = "business_analytics.db"
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Business Analytics Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
        h1 { color: #333; text-align: center; }
        .section { margin: 20px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { margin: 10px 0; }
        .alert { color: red; font-weight: bold; }
        .recommendation { color: green; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f2f2f2; }
        .trend-up { color: green; }
        .trend-down { color: red; }
    </style>
</head>
<body>
    <h1>Business Analytics Dashboard</h1>
    <div class="section">
        <h2>Financial Metrics</h2>
        <div class="metric">Revenue: ${{ financial_metrics.revenue.current }} ({{ financial_metrics.revenue.trend }})</div>
        <div class="metric">Cost: ${{ financial_metrics.cost.current }} ({{ financial_metrics.cost.trend }})</div>
        <div class="metric">Profit: ${{ financial_metrics.profit.amount }} ({{ financial_metrics.profit.status }})</div>
        <div class="metric">Profit Margin: {{ financial_metrics.profit.margin }}%</div>
    </div>
    <div class="section">
        <h2>Customer Metrics</h2>
        <div class="metric">Customer Count: {{ customer_metrics.count }}</div>
        <div class="metric">CAC: ${{ customer_metrics.cac.current }} ({{ customer_metrics.cac.status }})</div>
    </div>
    <div class="section">
        <h2>Insights</h2>
        {% if insights.alerts %}
            <h3>Alerts</h3>
            <ul>
                {% for alert in insights.alerts %}
                    <li class="alert">{{ alert }}</li>
                {% endfor %}
            </ul>
        {% endif %}
        {% if insights.recommendations %}
            <h3>Recommendations</h3>
            <ul>
                {% for rec in insights.recommendations %}
                    <li class="recommendation">{{ rec }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
    <div class="section">
        <h2>Trend Forecast</h2>
        <table>
            <tr><th>Metric</th><th>Current</th><th>Forecast (Next Day)</th></tr>
            <tr><td>Revenue</td><td>{{ financial_metrics.revenue.current }}</td><td>{{ trend_forecast.revenue }}</td></tr>
            <tr><td>Cost</td><td>{{ financial_metrics.cost.current }}</td><td>{{ trend_forecast.cost }}</td></tr>
            <tr><td>CAC</td><td>{{ customer_metrics.cac.current }}</td><td>{{ trend_forecast.cac }}</td></tr>
        </table>
    </div>
</body>
</html>
"""

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
    """
    Configuration settings for the business analytics agent.
    
    Attributes:
        output_dir (str): Directory for storing reports
        decimal_precision (int): Number of decimal places for rounding
        enable_backup (bool): Enable/disable backup functionality
        max_reports (int): Maximum number of reports to retain
        enable_db_storage (bool): Enable/disable SQLite storage
        forecast_horizon (int): Number of days for trend forecasting
        custom_thresholds (dict): Custom thresholds for alerts
    """
    output_dir: str = "reports"
    decimal_precision: int = 2
    enable_backup: bool = True
    max_reports: int = 30
    enable_db_storage: bool = True
    forecast_horizon: int = 1
    custom_thresholds: Dict[str, float] = None

    def __post_init__(self):
        """Initialize default custom thresholds if not provided."""
        if self.custom_thresholds is None:
            self.custom_thresholds = {
                'cac_increase': CAC_INCREASE_THRESHOLD,
                'revenue_growth': REVENUE_GROWTH_THRESHOLD,
                'profit_warning': PROFIT_WARNING_THRESHOLD
            }

class BusinessState(TypedDict):
    """
    Data structure for the analysis workflow state.
    
    Attributes:
        daily_revenue (float): Current day's revenue
        daily_cost (float): Current day's cost
        num_customers (int): Current day's customer count
        prev_day_revenue (float): Previous day's revenue
        prev_day_cost (float): Previous day's cost
        prev_day_customers (int): Previous day's customer count
        profit (Optional[float]): Calculated profit
        revenue_change_pct (Optional[float]): Revenue change percentage
        cost_change_pct (Optional[float]): Cost change percentage
        current_cac (Optional[float]): Current customer acquisition cost
        prev_cac (Optional[float]): Previous customer acquisition cost
        cac_change_pct (Optional[float]): CAC change percentage
        alerts (List[str]): List of generated alerts
        recommendations (List[str]): List of generated recommendations
        metadata (Dict[str, Any]): Metadata for the analysis
        financial_ratios (Dict[str, float]): Financial ratios
        trend_forecast (Dict[str, float]): Forecasted values
    """
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
    financial_ratios: Dict[str, float]
    trend_forecast: Dict[str, float]

class BusinessAnalyticsAgent:
    """
    Production-grade business analytics solution with advanced features.
    
    Attributes:
        VERSION (str): Agent version
        config (BusinessConfig): Configuration object
        workflow (StateGraph): Workflow graph for analysis
        output_dir (Path): Directory for report storage
        backup_dir (Path): Directory for backup storage
        db_conn (sqlite3.Connection): SQLite database connection
    """
    
    VERSION = "3.1.0"
    
    def __init__(self, config: BusinessConfig = BusinessConfig()):
        """
        Initialize the BusinessAnalyticsAgent.
        
        Args:
            config (BusinessConfig): Configuration object for the agent
        """
        self.config = config
        self.workflow = StateGraph(BusinessState)
        self._setup_environment()
        self._setup_database()
        self._build_graph()
        self.agent = self.workflow.compile()
        logger.info(f"BusinessAnalyticsAgent v{self.VERSION} initialized")

    def _setup_environment(self):
        """
        Initialize directories and environment for report storage.
        Creates output and backup directories if they don't exist.
        """
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if self.config.enable_backup:
            self.backup_dir = self.output_dir / "backups"
            self.backup_dir.mkdir(exist_ok=True)
        
        self._cleanup_old_reports()

    def _setup_database(self):
        """
        Initialize SQLite database for storing report metadata.
        Creates tables if they don't exist.
        """
        if self.config.enable_db_storage:
            try:
                self.db_conn = sqlite3.connect(DB_NAME)
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_filename TEXT,
                        data_hash TEXT,
                        analysis_date TEXT,
                        revenue REAL,
                        cost REAL,
                        profit REAL,
                        customers INTEGER,
                        cac REAL
                    )
                """)
                self.db_conn.commit()
                logger.info("Database initialized successfully")
            except sqlite3.Error as e:
                logger.error(f"Database initialization failed: {str(e)}")
                raise

    def _cleanup_old_reports(self):
        """
        Maintain report directory size by removing old reports.
        Keeps only the most recent reports up to max_reports limit.
        """
        reports = sorted(self.output_dir.glob("business_report_*.json"))
        if len(reports) > self.config.max_reports:
            for old_report in reports[:-self.config.max_reports]:
                try:
                    old_report.unlink()
                    logger.debug(f"Removed old report: {old_report.name}")
                except OSError as e:
                    logger.error(f"Failed to remove old report {old_report.name}: {str(e)}")

    def _build_graph(self):
        """
        Construct the analysis workflow graph.
        Adds nodes and edges for the analysis pipeline.
        """
        self.workflow.add_node("validate_input", self._validate_input_node)
        self.workflow.add_node("calculate_metrics", self._metrics_node)
        self.workflow.add_node("calculate_ratios", self._ratios_node)
        self.workflow.add_node("forecast_trends", self._forecast_node)
        self.workflow.add_node("generate_insights", self._insights_node)
        self.workflow.add_node("prepare_report", self._report_node)
        self.workflow.add_node("generate_dashboard", self._dashboard_node)
        
        self.workflow.add_edge("validate_input", "calculate_metrics")
        self.workflow.add_edge("calculate_metrics", "calculate_ratios")
        self.workflow.add_edge("calculate_ratios", "forecast_trends")
        self.workflow.add_edge("forecast_trends", "generate_insights")
        self.workflow.add_edge("generate_insights", "prepare_report")
        self.workflow.add_edge("prepare_report", "generate_dashboard")
        
        self.workflow.set_entry_point("validate_input")
        self.workflow.set_finish_point("generate_dashboard")

    def _validate_input_node(self, state: BusinessState) -> BusinessState:
        """
        Validate and sanitize input data.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            BusinessState: Updated state with validated data
        """
        required_fields = {
            'daily_revenue': (float, MIN_REVENUE, "Revenue must be a non-negative number"),
            'daily_cost': (float, MIN_REVENUE, "Cost must be a non-negative number"),
            'num_customers': (int, MIN_CUSTOMERS, "Customer count must be a positive integer"),
            'prev_day_revenue': (float, MIN_REVENUE, "Previous revenue must be a non-negative number"),
            'prev_day_cost': (float, MIN_REVENUE, "Previous cost must be a non-negative number")
        }
        
        for field, (field_type, min_value, error_msg) in required_fields.items():
            if field not in state:
                raise ValueError(f"Missing required field: {field}")
            
            try:
                state[field] = field_type(state[field])
                if state[field] < min_value:
                    logger.warning(f"Low value warning for {field}: {state[field]}")
            except (TypeError, ValueError) as e:
                raise ValueError(f"{error_msg}: {e}")

        state.setdefault('prev_day_customers', state['num_customers'])
        state['alerts'] = []
        state['recommendations'] = []
        state['financial_ratios'] = {}
        state['trend_forecast'] = {}
        state['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'agent_version': self.VERSION,
            'report_version': REPORT_VERSION,
            'data_hash': self._generate_data_hash(state)
        }
        
        return state

    def _generate_data_hash(self, state: BusinessState) -> str:
        """
        Generate a hash for input data validation.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            str: MD5 hash of the input data
        """
        hash_data = {
            'revenue': state['daily_revenue'],
            'cost': state['daily_cost'],
            'customers': state['num_customers'],
            'timestamp': datetime.now().timestamp()
        }
        return hashlib.sha256(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()

    def _metrics_node(self, state: BusinessState) -> BusinessState:
        """
        Calculate core business metrics.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            BusinessState: Updated state with calculated metrics
        """
        try:
            state['profit'] = self._calculate_profit(
                state['daily_revenue'],
                state['daily_cost']
            )
            
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

    def _ratios_node(self, state: BusinessState) -> BusinessState:
        """
        Calculate financial ratios.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            BusinessState: Updated state with financial ratios
        """
        try:
            state['financial_ratios'] = {
                'gross_margin': self._calculate_gross_margin(
                    state['daily_revenue'],
                    state['daily_cost']
                ),
                'net_profit_margin': self._calculate_profit_margin(
                    state['daily_revenue'],
                    state['profit']
                ),
                'customer_lifetime_value_ratio': self._calculate_clv_ratio(
                    state['current_cac'],
                    state['daily_revenue'],
                    state['num_customers']
                )
            }
            return state
        except Exception as e:
            logger.error(f"Financial ratios calculation failed: {str(e)}")
            raise

    def _forecast_node(self, state: BusinessState) -> BusinessState:
        """
        Forecast trends for key metrics.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            BusinessState: Updated state with forecasted values
        """
        try:
            state['trend_forecast'] = {
                'revenue': self._forecast_metric(
                    state['daily_revenue'],
                    state['prev_day_revenue'],
                    self.config.forecast_horizon
                ),
                'cost': self._forecast_metric(
                    state['daily_cost'],
                    state['prev_day_cost'],
                    self.config.forecast_horizon
                ),
                'cac': self._forecast_metric(
                    state['current_cac'],
                    state['prev_cac'],
                    self.config.forecast_horizon
                )
            }
            return state
        except Exception as e:
            logger.error(f"Trend forecasting failed: {str(e)}")
            raise

    def _insights_node(self, state: BusinessState) -> BusinessState:
        """
        Generate actionable business insights based on metrics and ratios.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            BusinessState: Updated state with insights
        """
        # Profit analysis
        if state['profit'] < self.config.custom_thresholds['profit_warning']:
            alert_msg = "ALERT: Negative profit detected"
            recommendation = "Implement cost reduction measures and explore revenue growth opportunities"
            state['alerts'].append(alert_msg)
            state['recommendations'].append(recommendation)
            logger.warning(alert_msg)
        
        # Revenue growth analysis
        if state['revenue_change_pct'] > self.config.custom_thresholds['revenue_growth']:
            recommendation = "Allocate additional budget to high-performing marketing channels"
            state['recommendations'].append(recommendation)
            logger.info("Significant revenue growth detected")
        
        # CAC analysis
        if state['cac_change_pct'] > self.config.custom_thresholds['cac_increase']:
            rounded_change = round(state['cac_change_pct'], self.config.decimal_precision)
            alert_msg = f"ALERT: CAC increased by {rounded_change}% (Threshold: {self.config.custom_thresholds['cac_increase']}%)"
            recommendation = "Audit marketing campaigns and optimize acquisition channels"
            state['alerts'].append(alert_msg)
            state['recommendations'].append(recommendation)
            logger.warning(alert_msg)
        
        # Financial ratios analysis
        if state['financial_ratios']['gross_margin'] < 20.0:
            alert_msg = "ALERT: Low gross margin detected"
            recommendation = "Review cost structure and pricing strategy"
            state['alerts'].append(alert_msg)
            state['recommendations'].append(recommendation)
            logger.warning(alert_msg)
        
        return state

    def _report_node(self, state: BusinessState) -> Dict[str, Any]:
        """
        Generate comprehensive business report.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            Dict[str, Any]: Generated report
        """
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
                        'margin': round(state['financial_ratios']['net_profit_margin'], self.config.decimal_precision),
                        'status': 'positive' if state['profit'] >= 0 else 'negative'
                    }
                },
                'customer_metrics': {
                    'count': state['num_customers'],
                    'cac': {
                        'current': round(state['current_cac'], self.config.decimal_precision),
                        'previous': round(state['prev_cac'], self.config.decimal_precision),
                        'change_pct': round(state['cac_change_pct'], self.config.decimal_precision),
                        'status': 'critical' if state['cac_change_pct'] > self.config.custom_thresholds['cac_increase'] else 'normal'
                    }
                },
                'financial_ratios': {
                    'gross_margin': round(state['financial_ratios']['gross_margin'], self.config.decimal_precision),
                    'net_profit_margin': round(state['financial_ratios']['net_profit_margin'], self.config.decimal_precision),
                    'clv_ratio': round(state['financial_ratios']['customer_lifetime_value_ratio'], self.config.decimal_precision)
                },
                'trend_forecast': {
                    'revenue': round(state['trend_forecast']['revenue'], self.config.decimal_precision),
                    'cost': round(state['trend_forecast']['cost'], self.config.decimal_precision),
                    'cac': round(state['trend_forecast']['cac'], self.config.decimal_precision)
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
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"business_report_{timestamp}.json"
            self._save_report(report, report_filename)
            
            if self.config.enable_backup:
                backup_filename = f"backup_{report_filename}"
                self._save_report(report, backup_filename, backup=True)
            
            if self.config.enable_db_storage:
                self._store_report_in_db(report, report_filename)
            
            logger.info(f"Report generated: {report_filename}")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    def _dashboard_node(self, state: BusinessState) -> Dict[str, Any]:
        """
        Generate HTML dashboard for the report.
        
        Args:
            state (BusinessState): Current state of the analysis
        
        Returns:
            Dict[str, Any]: Updated report with dashboard information
        """
        try:
            report = self._report_node(state)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_filename = f"dashboard_{timestamp}.html"
            
            template = Template(HTML_TEMPLATE)
            dashboard_content = template.render(**report)
            
            with open(self.output_dir / dashboard_filename, 'w') as f:
                f.write(dashboard_content)
            
            report['dashboard'] = {'filename': dashboard_filename}
            logger.info(f"Dashboard generated: {dashboard_filename}")
            return report
        except Exception as e:
            logger.error(f"Dashboard generation failed: {str(e)}")
            raise

    def _save_report(self, data: Dict, filename: str, backup: bool = False):
        """
        Save report to JSON file with validation.
        
        Args:
            data (Dict): Report data to save
            filename (str): Name of the file
            backup (bool): Whether to save in backup directory
        """
        try:
            target_dir = self.backup_dir if backup else self.output_dir
            filepath = target_dir / filename
            
            self._validate_report_data(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            raise

    def _store_report_in_db(self, report: Dict, filename: str):
        """
        Store report metadata in SQLite database.
        
        Args:
            report (Dict): Report data
            filename (str): Report filename
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO reports (
                    report_filename, data_hash, analysis_date, 
                    revenue, cost, profit, customers, cac
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                report['metadata']['data_hash'],
                report['metadata']['analysis_date'],
                report['financial_metrics']['revenue']['current'],
                report['financial_metrics']['cost']['current'],
                report['financial_metrics']['profit']['amount'],
                report['customer_metrics']['count'],
                report['customer_metrics']['cac']['current']
            ))
            self.db_conn.commit()
            logger.debug(f"Report metadata stored in database for {filename}")
        except sqlite3.Error as e:
            logger.error(f"Failed to store report in database: {str(e)}")
            raise

    def _validate_report_data(self, data: Dict):
        """
        Validate report data structure.
        
        Args:
            data (Dict): Report data to validate
        """
        required_sections = ['metadata', 'financial_metrics', 'customer_metrics', 'insights', 'financial_ratios', 'trend_forecast']
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Invalid report format: missing {section} section")

    @staticmethod
    def _calculate_profit(revenue: float, cost: float) -> float:
        """
        Calculate daily profit.
        
        Args:
            revenue (float): Daily revenue
            cost (float): Daily cost
        
        Returns:
            float: Calculated profit
        """
        return revenue - cost

    @staticmethod
    def _calculate_profit_margin(revenue: float, profit: float) -> float:
        """
        Calculate profit margin percentage.
        
        Args:
            revenue (float): Daily revenue
            profit (float): Daily profit
        
        Returns:
            float: Profit margin percentage
        """
        if revenue == 0:
            return 0.0
        return (profit / revenue) * 100

    @staticmethod
    def _calculate_gross_margin(revenue: float, cost: float) -> float:
        """
        Calculate gross margin percentage.
        
        Args:
            revenue (float): Daily revenue
            cost (float): Daily cost
        
        Returns:
            float: Gross margin percentage
        """
        if revenue == 0:
            return 0.0
        return ((revenue - cost) / revenue) * 100

    @staticmethod
    def _calculate_clv_ratio(cac: float, revenue: float, customers: int) -> float:
        """
        Calculate customer lifetime value to CAC ratio.
        
        Args:
            cac (float): Customer acquisition cost
            revenue (float): Daily revenue
            customers (int): Number of customers
        
        Returns:
            float: CLV to CAC ratio
        """
        if customers == 0 or cac == 0:
            return 0.0
        return (revenue / customers) / cac

    @staticmethod
    def _calculate_change_percentage(current: float, previous: float, metric_name: str = "") -> float:
        """
        Calculate percentage change with validation.
        
        Args:
            current (float): Current value
            previous (float): Previous value
            metric_name (str): Name of the metric
        
        Returns:
            float: Percentage change
        """
        if previous == 0:
            logger.warning(f"Cannot calculate percentage change for {metric_name} - previous value is zero")
            return float('inf') if current > 0 else float('-inf')
        return ((current - previous) / previous) * 100

    @staticmethod
    def _calculate_cac(cost: float, customers: int) -> float:
        """
        Calculate Customer Acquisition Cost.
        
        Args:
            cost (float): Daily cost
            customers (int): Number of customers
        
        Returns:
            float: Customer Acquisition Cost
        """
        if customers < MIN_CUSTOMERS:
            logger.warning(f"Customer count below minimum ({MIN_CUSTOMERS})")
            return float('inf')
        return cost / customers

    @staticmethod
    def _forecast_metric(current: float, previous: float, horizon: int) -> float:
        """
        Forecast metric value using simple linear extrapolation.
        
        Args:
            current (float): Current value
            previous (float): Previous value
            horizon (int): Forecast horizon in days
        
        Returns:
            float: Forecasted value
        """
        if previous == 0:
            return current
        trend = (current - previous) / previous
        return current * (1 + trend * horizon)

    def analyze(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full analysis workflow.
        
        Args:
            business_data (Dict[str, Any]): Input business data
        
        Returns:
            Dict[str, Any]: Analysis report
        """
        try:
            logger.info("Starting business analysis")
            result = self.agent.invoke(business_data)
            logger.info("Analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    def test_analysis(self) -> Dict[str, Any]:
        """
        Execute comprehensive test cases.
        
        Returns:
            Dict[str, Any]: Test report
        """
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
                    f"ALERT: CAC increased by 44.64% (Threshold: {self.config.custom_thresholds['cac_increase']}%)",
                    "ALERT: Low gross margin detected"
                ],
                "expected_recommendations": [
                    "Implement cost reduction measures and explore revenue growth opportunities",
                    "Audit marketing campaigns and optimize acquisition channels",
                    "Review cost structure and pricing strategy"
                ]
            },
            {
                "name": "Stable business scenario",
                "input": {
                    "daily_revenue": 10000,
                    "daily_cost": 6000,
                    "num_customers": 100,
                    "prev_day_revenue": 10000,
                    "prev_day_cost": 6000,
                    "prev_day_customers": 100
                },
                "expected_alerts": [],
                "expected_recommendations": []
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
        
        test_report = {
            "summary": {
                "total_tests": len(test_cases),
                "passed": sum(1 for r in test_results if r["status"] == "passed"),
                "failed": sum(1 for r in test_results if r["status"] == "failed"),
                "success_rate": f"{sum(1 for r in test_results if r['status'] == 'passed') / len(test_cases) * 100:.1f}%"
            },
            "details": test_results
        }
        
        test_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_report(test_report, test_filename)
        
        return test_report

def main():
    """
    Entry point for the application.
    Runs sample analysis and test suite.
    """
    try:
        config = BusinessConfig(
            output_dir="business_reports",
            decimal_precision=2,
            enable_backup=True,
            max_reports=30,
            enable_db_storage=True,
            forecast_horizon=1,
            custom_thresholds={
                'cac_increase': 20.0,
                'revenue_growth': 10.0,
                'profit_warning': 0.0
            }
        )
        
        agent = BusinessAnalyticsAgent(config)
        logger.info("Business Analytics Agent initialized successfully")
        
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
            "recommendations": report["insights"]["recommendations"],
            "dashboard": report["dashboard"]
        }, indent=2))
        
        print("\nRunning test suite...")
        test_results = agent.test_analysis()
        print("\nTest Results Summary:")
        print(json.dumps(test_results["summary"], indent=2))
        
    except Exception as e:
        logger.critical(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()