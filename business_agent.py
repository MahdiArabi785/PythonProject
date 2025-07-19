from langgraph.graph import StateGraph
from typing import Dict, TypedDict, List, Optional, Any
import json


class BusinessState(TypedDict):
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
    recommendations: List[str]
    alerts: List[str]


class BusinessAnalyticsAgent:
    """A business analytics agent that analyzes daily sales data"""

    def __init__(self):
        self.workflow = StateGraph(BusinessState)
        self._build_graph()
        self.agent = self.workflow.compile()

    def _build_graph(self):
        """Construct the analysis workflow"""
        self.workflow.add_node("input", self._input_node)
        self.workflow.add_node("process", self._processing_node)
        self.workflow.add_node("recommend", self._recommendation_node)
        self.workflow.add_node("output", self._output_node)

        self.workflow.add_edge("input", "process")
        self.workflow.add_edge("process", "recommend")
        self.workflow.add_edge("recommend", "output")

        self.workflow.set_entry_point("input")
        self.workflow.set_finish_point("output")

    def _input_node(self, state: BusinessState) -> BusinessState:
        """Validate and prepare input data"""
        required_fields = ['daily_revenue', 'daily_cost', 'num_customers',
                           'prev_day_revenue', 'prev_day_cost']
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field: {field}")

        state.setdefault('prev_day_customers', state['num_customers'])
        state['recommendations'] = []
        state['alerts'] = []
        return state

    def _processing_node(self, state: BusinessState) -> BusinessState:
        """Calculate key business metrics"""
        state['profit'] = state['daily_revenue'] - state['daily_cost']

        state['revenue_change_pct'] = self._calculate_percentage_change(
            state['daily_revenue'], state['prev_day_revenue'])

        state['cost_change_pct'] = self._calculate_percentage_change(
            state['daily_cost'], state['prev_day_cost'])

        state['current_cac'] = self._calculate_cac(
            state['daily_cost'], state['num_customers'])

        state['prev_cac'] = self._calculate_cac(
            state['prev_day_cost'], state['prev_day_customers'])

        state['cac_change_pct'] = self._calculate_percentage_change(
            state['current_cac'], state['prev_cac'])

        return state

    def _recommendation_node(self, state: BusinessState) -> BusinessState:
        """Generate business recommendations"""
        SIGNIFICANT_GROWTH = 10  # 10% revenue growth
        SIGNIFICANT_DECLINE = -10
        CAC_INCREASE_WARNING = 20  # 20% CAC increase as per requirements

        # Profit analysis
        if state['profit'] < 0:
            state['alerts'].append("ALERT: Negative profit detected")
            state['recommendations'].append("Reduce costs or increase revenue to improve profitability")

        # Revenue change analysis
        if state['revenue_change_pct'] > SIGNIFICANT_GROWTH:
            state['recommendations'].append("Consider increasing advertising budget as sales are growing")
        elif state['revenue_change_pct'] < SIGNIFICANT_DECLINE:
            state['alerts'].append("ALERT: Significant revenue drop detected")
            state['recommendations'].append("Investigate reasons for revenue decline")

        # CAC analysis
        if state['cac_change_pct'] > CAC_INCREASE_WARNING:
            cac_change = round(state['cac_change_pct'], 2)
            state['alerts'].append(f"ALERT: CAC increased by {cac_change}%")
            state['recommendations'].append("Review marketing campaigns - CAC increased significantly")

        return state

    def _output_node(self, state: BusinessState) -> Dict[str, Any]:
        """Format the final output"""
        return {
            'profit_status': 'positive' if state['profit'] >= 0 else 'negative',
            'profit_amount': state['profit'],
            'revenue_change_percentage': state['revenue_change_pct'],
            'cost_change_percentage': state['cost_change_pct'],
            'customer_acquisition_cost': state['current_cac'],
            'alerts': state['alerts'],
            'recommendations': state['recommendations']
        }

    @staticmethod
    def _calculate_percentage_change(current: float, previous: float) -> float:
        """Calculate percentage change between two values"""
        if previous == 0:
            return float('inf')
        return ((current - previous) / previous) * 100

    @staticmethod
    def _calculate_cac(cost: float, customers: int) -> float:
        """Calculate Customer Acquisition Cost"""
        if customers <= 0:
            return float('inf')
        return cost / customers

    def analyze(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analysis on input business data"""
        return self.agent.invoke(business_data)


if __name__ == "__main__":
    # Example usage
    agent = BusinessAnalyticsAgent()

    # Sample data
    business_data = {
        "daily_revenue": 12000,
        "daily_cost": 8500,
        "num_customers": 95,
        "prev_day_revenue": 10000,
        "prev_day_cost": 8000,
        "prev_day_customers": 90
    }

    # Run analysis
    result = agent.analyze(business_data)
    print("Business Analysis Report:")
    print(json.dumps(result, indent=2))