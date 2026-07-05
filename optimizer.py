"""
Charity Fundraising Campaign Optimizer - Optimization Engine
Implements budget allocation, campaign selection, and resource optimization.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class CampaignOptimizer:
    """Optimization engine for charity fundraising campaigns."""

    def __init__(self):
        self.optimization_history = []

    def optimize_budget_allocation(self, campaigns, total_budget, method='roi_maximization'):
        """
        Optimize budget allocation across multiple campaigns.

        Args:
            campaigns: DataFrame with campaign options and predicted ROI
            total_budget: Total available budget
            method: 'roi_maximization', 'risk_adjusted', 'diversified'
        """
        n = len(campaigns)

        if method == 'roi_maximization':
            # Maximize total expected return
            # Objective: maximize sum(roi_i * budget_i)
            # Constraint: sum(budget_i) <= total_budget

            predicted_rois = campaigns['predicted_roi'].values

            # Simple greedy approach with minimum allocation
            min_budget = total_budget * 0.05  # 5% minimum per campaign

            # Sort by ROI
            sorted_idx = np.argsort(predicted_rois)[::-1]
            allocation = np.zeros(n)
            remaining = total_budget

            for idx in sorted_idx:
                if remaining <= 0:
                    break
                alloc = min(remaining * 0.4, remaining)  # Max 40% to one campaign
                alloc = max(alloc, min_budget) if remaining >= min_budget else alloc
                allocation[idx] = alloc
                remaining -= alloc

            # Normalize to total budget
            if allocation.sum() > 0:
                allocation = allocation / allocation.sum() * total_budget

        elif method == 'risk_adjusted':
            # Use predicted ROI and confidence to balance risk
            predicted_rois = campaigns['predicted_roi'].values
            confidence = campaigns.get('confidence_score', pd.Series([0.5]*n)).values

            # Risk-adjusted score = ROI * confidence
            scores = predicted_rois * confidence
            sorted_idx = np.argsort(scores)[::-1]

            allocation = np.zeros(n)
            remaining = total_budget

            for idx in sorted_idx:
                if remaining <= 0:
                    break
                weight = scores[idx] / scores.sum() if scores.sum() > 0 else 1/n
                alloc = total_budget * weight
                alloc = min(alloc, remaining * 0.5)
                allocation[idx] = alloc
                remaining -= alloc

            if allocation.sum() > 0:
                allocation = allocation / allocation.sum() * total_budget

        elif method == 'diversified':
            # Diversify across campaign types
            campaign_types = campaigns['campaign_type'].unique()
            allocation = np.zeros(n)

            budget_per_type = total_budget / len(campaign_types)

            for ctype in campaign_types:
                type_mask = campaigns['campaign_type'] == ctype
                type_campaigns = campaigns[type_mask]

                if len(type_campaigns) > 0:
                    # Allocate within type based on ROI
                    rois = type_campaigns['predicted_roi'].values
                    if rois.sum() > 0:
                        weights = rois / rois.sum()
                    else:
                        weights = np.ones(len(rois)) / len(rois)

                    allocation[type_mask] = budget_per_type * weights

        campaigns = campaigns.copy()
        campaigns['optimized_budget'] = allocation
        campaigns['expected_return'] = allocation * campaigns['predicted_roi']
        campaigns['expected_amount_raised'] = allocation * (1 + campaigns['predicted_roi'])

        return campaigns

    def select_optimal_campaigns(self, options, budget, max_campaigns=5, min_roi=0.3):
        """
        Select the best combination of campaigns within budget.
        Uses combinatorial optimization.
        """
        best_value = 0
        best_selection = None

        # Filter options with minimum ROI
        viable = options[options['predicted_roi'] >= min_roi].copy()

        if len(viable) == 0:
            return pd.DataFrame()

        # Try different numbers of campaigns
        for k in range(1, min(max_campaigns + 1, len(viable) + 1)):
            for combo in combinations(viable.index, k):
                selected = viable.loc[list(combo)]
                total_cost = selected['budget'].sum()

                if total_cost <= budget:
                    total_value = selected['predicted_roi'].sum()
                    if total_value > best_value:
                        best_value = total_value
                        best_selection = selected.copy()

        if best_selection is not None:
            best_selection['selection_rank'] = range(1, len(best_selection) + 1)

        return best_selection if best_selection is not None else pd.DataFrame()

    def optimize_campaign_timing(self, campaigns, months_ahead=12):
        """
        Optimize when to run campaigns throughout the year.
        Considers seasonality and donor fatigue.
        """
        # Seasonality multipliers (based on fundraising patterns)
        seasonality = {
            1: 0.8,   # January - post-holiday slump
            2: 0.9,
            3: 1.0,
            4: 1.0,
            5: 1.1,
            6: 0.9,   # Summer
            7: 0.8,   # Summer
            8: 0.9,
            9: 1.1,   # Back to school
            10: 1.2,  # Fall giving season
            11: 1.4,  # Giving Tuesday, year-end
            12: 1.5   # Year-end giving
        }

        results = []
        for _, campaign in campaigns.iterrows():
            best_month = max(seasonality.keys(), 
                           key=lambda m: seasonality[m] * (1 if m != campaign.get('proposed_month', m) else 1.1))

            results.append({
                'campaign_id': campaign.get('option_id', campaign.get('campaign_id')),
                'campaign_type': campaign['campaign_type'],
                'original_month': campaign.get('proposed_month', 'N/A'),
                'recommended_month': best_month,
                'seasonality_boost': seasonality[best_month],
                'expected_roi_multiplier': round(seasonality[best_month], 2)
            })

        return pd.DataFrame(results)

    def donor_segment_strategy(self, segments_df):
        """
        Generate targeted strategies for each donor segment.
        """
        strategies = []

        for segment in segments_df['segment_name'].unique():
            seg_data = segments_df[segments_df['segment_name'] == segment]

            if segment == 'Champions':
                strategy = {
                    'segment': segment,
                    'size': len(seg_data),
                    'avg_donation': seg_data['avg_monetary'].mean(),
                    'recommended_approach': 'Major donor cultivation',
                    'campaign_types': ['Major Donor Event', 'Personalized Appeals', 'Matching Gift'],
                    'frequency': 'Monthly personal touchpoints',
                    'expected_response_rate': 0.45,
                    'priority': 'Highest'
                }
            elif segment == 'Loyal Supporters':
                strategy = {
                    'segment': segment,
                    'size': len(seg_data),
                    'avg_donation': seg_data['avg_monetary'].mean(),
                    'recommended_approach': 'Retention & upgrade',
                    'campaign_types': ['Monthly Giving', 'Peer-to-Peer', 'Event Fundraising'],
                    'frequency': 'Quarterly campaigns',
                    'expected_response_rate': 0.30,
                    'priority': 'High'
                }
            elif segment == 'Potential Donors':
                strategy = {
                    'segment': segment,
                    'size': len(seg_data),
                    'avg_donation': seg_data['avg_monetary'].mean(),
                    'recommended_approach': 'Engagement & conversion',
                    'campaign_types': ['Email Campaign', 'Social Media', 'Crowdfunding'],
                    'frequency': 'Bi-weekly engagement',
                    'expected_response_rate': 0.15,
                    'priority': 'Medium'
                }
            else:  # At-Risk Donors
                strategy = {
                    'segment': segment,
                    'size': len(seg_data),
                    'avg_donation': seg_data['avg_monetary'].mean(),
                    'recommended_approach': 'Win-back campaign',
                    'campaign_types': ['Direct Mail', 'Phone Campaign', 'Special Appeal'],
                    'frequency': 'Immediate intervention',
                    'expected_response_rate': 0.08,
                    'priority': 'Urgent'
                }

            strategies.append(strategy)

        return pd.DataFrame(strategies)

    def calculate_portfolio_metrics(self, campaigns_df):
        """Calculate overall portfolio performance metrics."""
        total_budget = campaigns_df['optimized_budget'].sum()
        expected_return = campaigns_df['expected_return'].sum()
        expected_amount = campaigns_df['expected_amount_raised'].sum()

        # Portfolio ROI
        portfolio_roi = expected_return / total_budget if total_budget > 0 else 0

        # Risk metrics
        roi_values = campaigns_df['predicted_roi'].values
        roi_std = np.std(roi_values)
        sharpe_ratio = portfolio_roi / roi_std if roi_std > 0 else 0

        # Concentration risk
        max_allocation = campaigns_df['optimized_budget'].max() / total_budget if total_budget > 0 else 0

        return {
            'total_budget': total_budget,
            'expected_return': expected_return,
            'expected_amount_raised': expected_amount,
            'portfolio_roi': portfolio_roi,
            'roi_volatility': roi_std,
            'sharpe_ratio': sharpe_ratio,
            'max_concentration': max_allocation,
            'num_campaigns': len(campaigns_df),
            'diversification_score': 1 - max_allocation
        }

    def generate_optimization_report(self, campaigns_df, portfolio_metrics, timing_df=None, donor_strategies=None):
        """Generate comprehensive optimization report."""
        report = []
        report.append("=" * 70)
        report.append("CHARITY FUNDRAISING CAMPAIGN OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Portfolio Summary
        report.append("📊 PORTFOLIO SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Budget Allocated:    ${portfolio_metrics['total_budget']:,.2f}")
        report.append(f"Expected Return:           ${portfolio_metrics['expected_return']:,.2f}")
        report.append(f"Expected Amount Raised:    ${portfolio_metrics['expected_amount_raised']:,.2f}")
        report.append(f"Portfolio ROI:             {portfolio_metrics['portfolio_roi']:.2%}")
        report.append(f"ROI Volatility:            {portfolio_metrics['roi_volatility']:.4f}")
        report.append(f"Sharpe Ratio:              {portfolio_metrics['sharpe_ratio']:.4f}")
        report.append(f"Diversification Score:     {portfolio_metrics['diversification_score']:.2%}")
        report.append(f"Number of Campaigns:       {portfolio_metrics['num_campaigns']}")
        report.append("")

        # Campaign Allocation
        report.append("📋 CAMPAIGN ALLOCATION")
        report.append("-" * 40)
        for _, row in campaigns_df.iterrows():
            report.append(f"  {row.get('campaign_type', 'Campaign')}")
            report.append(f"    Budget: ${row['optimized_budget']:,.2f} | "
                        f"Expected ROI: {row['predicted_roi']:.2%} | "
                        f"Expected Return: ${row['expected_return']:,.2f}")
        report.append("")

        # Timing Recommendations
        if timing_df is not None and len(timing_df) > 0:
            report.append("📅 TIMING OPTIMIZATION")
            report.append("-" * 40)
            for _, row in timing_df.iterrows():
                report.append(f"  {row['campaign_type']}: "
                            f"Month {row['recommended_month']} "
                            f"(Boost: {row['seasonality_boost']:.2f}x)")
            report.append("")

        # Donor Strategies
        if donor_strategies is not None and len(donor_strategies) > 0:
            report.append("👥 DONOR SEGMENT STRATEGIES")
            report.append("-" * 40)
            for _, row in donor_strategies.iterrows():
                report.append(f"  {row['segment']} ({row['size']} donors)")
                report.append(f"    Approach: {row['recommended_approach']}")
                report.append(f"    Priority: {row['priority']} | "
                            f"Expected Response: {row['expected_response_rate']:.1%}")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)


if __name__ == '__main__':
    # Test the optimizer
    optimizer = CampaignOptimizer()

    # Sample campaign data
    sample_campaigns = pd.DataFrame({
        'campaign_type': ['Email Campaign', 'Social Media', 'Event Fundraising', 'Direct Mail'],
        'predicted_roi': [2.5, 1.8, 3.2, 1.5],
        'confidence_score': [0.85, 0.75, 0.70, 0.90],
        'budget': [5000, 3000, 15000, 8000]
    })

    result = optimizer.optimize_budget_allocation(sample_campaigns, 50000, method='roi_maximization')
    print(result)
