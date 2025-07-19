from langgraph.graph import StateGraph
from typing import TypedDict, Dict, List, Optional, Any
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
    alerts: List[str]
    recommendations: List[str]

class BusinessAnalyticsAgent:
    """AI Agent for analyzing business data and generating recommendations"""
    
    def __init__(self):
        self.workflow = StateGraph(BusinessState)
        self._build_graph()
        self.agent = self.workflow.compile()
    
    def _build_graph(self):
        """Construct the analysis workflow graph"""
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
        required_fields = [
            'daily_revenue', 'daily_cost', 'num_customers',
            'prev_day_revenue', 'prev_day_cost'
        ]
        
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required field: {field}")
        
        state.setdefault('prev_day_customers', state['num_customers'])
        state['alerts'] = []
        state['recommendations'] = []
        
        return state
    
    def _processing_node(self, state: BusinessState) -> BusinessState:
        """Calculate key business metrics"""
        state['profit'] = state['daily_revenue'] - state['daily_cost']
        
        state['revenue_change_pct'] = self._calculate_percentage_change(
            current=state['daily_revenue'],
            previous=state['prev_day_revenue']
        )
        
        state['cost_change_pct'] = self._calculate_percentage_change(
            current=state['daily_cost'],
            previous=state['prev_day_cost']
        )
        
        state['current_cac'] = self._calculate_cac(
            cost=state['daily_cost'],
            customers=state['num_customers']
        )
        
        state['prev_cac'] = self._calculate_cac(
            cost=state['prev_day_cost'],
            customers=state['prev_day_customers']
        )
        
        state['cac_change_pct'] = self._calculate_percentage_change(
            current=state['current_cac'],
            previous=state['prev_cac']
        )
        
        return state
    
    def _recommendation_node(self, state: BusinessState) -> BusinessState:
        """Generate business recommendations based on metrics"""
        # Profit analysis
        if state['profit'] < 0:
            state['alerts'].append("ALERT: Negative profit detected")
            state['recommendations'].append("Reduce costs to improve profitability")
        
        # Revenue growth analysis
        if state['revenue_change_pct'] > 10:  # Significant growth threshold
            state['recommendations'].append(
                "Consider increasing advertising budget as sales are growing"
            )
        
        # CAC analysis
        if state['cac_change_pct'] > 20:  # Significant CAC increase threshold
            rounded_change = round(state['cac_change_pct'], 2)
            state['alerts'].append(
                f"ALERT: Customer Acquisition Cost increased by {rounded_change}%"
            )
            state['recommendations'].append(
                "Review marketing campaigns as CAC increased significantly"
            )
        
        return state
    
    def _output_node(self, state: BusinessState) -> Dict[str, Any]:
        """Format the final output report"""
        return {
            'profit_analysis': {
                'status': 'positive' if state['profit'] >= 0 else 'negative',
                'amount': state['profit']
            },
            'performance_metrics': {
                'revenue_change': f"{state['revenue_change_pct']:.2f}%",
                'cost_change': f"{state['cost_change_pct']:.2f}%",
                'current_cac': state['current_cac'],
                'cac_change': f"{state['cac_change_pct']:.2f}%"
            },
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
        """Calculate Customer Acquisition Cost (CAC)"""
        if customers <= 0:
            return float('inf')
        return cost / customers
    
    def analyze(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the analysis on input business data"""
        return self.agent.invoke(business_data)
    
    def test_agent(self):
        """Test the agent with sample data"""
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
                    "Consider increasing advertising budget as sales are growing"
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
                    "ALERT: Customer Acquisition Cost increased by 44.64%"
                ],
                "expected_recommendations": [
                    "Reduce costs to improve profitability",
                    "Review marketing campaigns as CAC increased significantly"
                ]
            }
        ]
        
        for case in test_cases:
            print(f"\nRunning test case: {case['name']}")
            result = self.analyze(case["input"])
            
            print("Agent output:")
            print(json.dumps(result, indent=2))
            
            assert set(case["expected_alerts"]) == set(result["alerts"]), (
                f"Alerts mismatch in {case['name']}\n"
                f"Expected: {case['expected_alerts']}\n"
                f"Got: {result['alerts']}"
            )
            
            assert set(case["expected_recommendations"]) == set(result["recommendations"]), (
                f"Recommendations mismatch in {case['name']}\n"
                f"Expected: {case['expected_recommendations']}\n"
                f"Got: {result['recommendations']}"
            )
            
            print("Test passed!")
        
        print("\nAll tests passed successfully!")

if __name__ == "__main__":
    agent = BusinessAnalyticsAgent()
    
    # Sample analysis
    sample_data = {
        "daily_revenue": 12000,
        "daily_cost": 8500,
        "num_customers": 95,
        "prev_day_revenue": 10000,
        "prev_day_cost": 8000,
        "prev_day_customers": 90
    }
    
    print("Running business analysis...")
    result = agent.analyze(sample_data)
    print("\nBusiness Analysis Report:")
    print(json.dumps(result, indent=2))
    
    # Run tests
    print("\nRunning test cases...")
    agent.test_agent()