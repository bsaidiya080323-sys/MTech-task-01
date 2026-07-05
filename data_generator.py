"""
Charity Fundraising Campaign Optimizer - Data Generator
Generates realistic sample data for campaigns, donors, and historical performance.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class CampaignDataGenerator:
    """Generates realistic charity fundraising campaign data."""

    def __init__(self, random_state=42):
        np.random.seed(random_state)
        random.seed(random_state)

        self.campaign_types = [
            'Email Campaign', 'Social Media', 'Direct Mail', 
            'Phone Campaign', 'Event Fundraising', 'Corporate Partnership',
            'Peer-to-Peer', 'Crowdfunding', 'Monthly Giving',
            'Emergency Appeal', 'Year-End Appeal', 'Matching Gift'
        ]

        self.channels = [
            'Email', 'Facebook', 'Instagram', 'Twitter', 'LinkedIn',
            'Direct Mail', 'Phone', 'Website', 'Events', 'SMS'
        ]

        self.target_audiences = [
            'Young Professionals', 'Retirees', 'Families', 'Students',
            'Corporate Donors', 'Major Donors', 'Recurring Donors',
            'First-Time Donors', 'Lapsed Donors', 'Volunteers'
        ]

        self.causes = [
            'Education', 'Healthcare', 'Disaster Relief', 'Animal Welfare',
            'Environment', 'Poverty Alleviation', 'Arts & Culture',
            'Community Development', 'Children Welfare', 'Elderly Care'
        ]

    def generate_historical_campaigns(self, n_campaigns=500):
        """Generate historical campaign data for training ML models."""
        data = []

        for i in range(n_campaigns):
            campaign_type = random.choice(self.campaign_types)
            channel = random.choice(self.channels)
            audience = random.choice(self.target_audiences)
            cause = random.choice(self.causes)

            # Budget ranges based on campaign type
            if campaign_type in ['Corporate Partnership', 'Major Donor Event']:
                budget = np.random.uniform(10000, 100000)
            elif campaign_type in ['Email Campaign', 'Social Media']:
                budget = np.random.uniform(500, 5000)
            else:
                budget = np.random.uniform(1000, 20000)

            # Duration in days
            duration = np.random.randint(7, 90)

            # Number of outreach attempts
            outreach_count = np.random.randint(1, 20)

            # Historical engagement rate (affects outcome)
            base_engagement = np.random.uniform(0.02, 0.25)

            # Campaign timing (seasonality factor)
            month = random.randint(1, 12)
            seasonality = 1.0
            if month in [11, 12]:  # Year-end giving
                seasonality = 1.4
            elif month in [6, 7]:  # Summer slump
                seasonality = 0.7

            # Calculate outcome with realistic relationships
            # Higher budget, better engagement, good timing = better results
            noise = np.random.normal(0, 0.1)

            # Amount raised depends on budget, engagement, duration, seasonality
            efficiency = base_engagement * seasonality * (1 + np.log1p(outreach_count) * 0.1)
            amount_raised = budget * efficiency * np.random.uniform(1.5, 5.0) * (1 + noise)
            amount_raised = max(budget * 0.5, amount_raised)  # At least 50% of budget

            # Number of donors
            avg_donation = np.random.uniform(25, 200)
            num_donors = int(amount_raised / avg_donation)

            # ROI calculation
            roi = (amount_raised - budget) / budget

            # Success classification
            if roi > 2.0:
                success_label = 'Highly Successful'
            elif roi > 1.0:
                success_label = 'Successful'
            elif roi > 0.3:
                success_label = 'Moderate'
            else:
                success_label = 'Underperformed'

            data.append({
                'campaign_id': f'CAMP_{i+1:04d}',
                'campaign_type': campaign_type,
                'channel': channel,
                'target_audience': audience,
                'cause': cause,
                'budget': round(budget, 2),
                'duration_days': duration,
                'outreach_count': outreach_count,
                'month': month,
                'engagement_rate': round(base_engagement, 4),
                'amount_raised': round(amount_raised, 2),
                'num_donors': num_donors,
                'avg_donation': round(avg_donation, 2),
                'roi': round(roi, 4),
                'success_label': success_label
            })

        return pd.DataFrame(data)

    def generate_donor_profiles(self, n_donors=1000):
        """Generate donor profile data."""
        data = []

        for i in range(n_donors):
            age = np.random.randint(18, 80)

            # Income based on age
            if age < 25:
                income = np.random.uniform(15000, 40000)
            elif age < 35:
                income = np.random.uniform(30000, 80000)
            elif age < 50:
                income = np.random.uniform(50000, 150000)
            else:
                income = np.random.uniform(40000, 200000)

            # Donation history
            years_donating = np.random.randint(0, 20)
            total_donations = np.random.exponential(500) * (years_donating + 1)

            # Preferred cause (biased by age)
            if age < 30:
                preferred_cause = random.choice(['Environment', 'Education', 'Animal Welfare'])
            elif age < 50:
                preferred_cause = random.choice(['Education', 'Healthcare', 'Community Development'])
            else:
                preferred_cause = random.choice(['Healthcare', 'Children Welfare', 'Elderly Care'])

            # Engagement score
            engagement_score = np.random.beta(2, 5) * 100

            # Recency (months since last donation)
            recency = np.random.exponential(6)

            # Frequency (donations per year)
            frequency = np.random.poisson(2) + 1

            # Monetary value
            monetary = total_donations / max(years_donating, 1)

            # RFM Score
            rfm_score = (1 / (1 + recency/12)) * 30 + min(frequency, 10) * 5 + np.log1p(monetary) * 5

            # Churn risk
            if recency > 12:
                churn_risk = 'High'
            elif recency > 6:
                churn_risk = 'Medium'
            else:
                churn_risk = 'Low'

            data.append({
                'donor_id': f'DONOR_{i+1:05d}',
                'age': age,
                'income': round(income, 2),
                'years_donating': years_donating,
                'total_donations': round(total_donations, 2),
                'preferred_cause': preferred_cause,
                'engagement_score': round(engagement_score, 2),
                'recency_months': round(recency, 2),
                'frequency_per_year': frequency,
                'avg_monetary': round(monetary, 2),
                'rfm_score': round(rfm_score, 2),
                'churn_risk': churn_risk
            })

        return pd.DataFrame(data)

    def generate_current_campaign_options(self, n_options=20):
        """Generate campaign options for optimization."""
        data = []

        for i in range(n_options):
            data.append({
                'option_id': f'OPT_{i+1:03d}',
                'campaign_type': random.choice(self.campaign_types),
                'channel': random.choice(self.channels),
                'target_audience': random.choice(self.target_audiences),
                'cause': random.choice(self.causes),
                'budget': round(np.random.uniform(1000, 50000), 2),
                'duration_days': np.random.randint(14, 60),
                'outreach_count': np.random.randint(3, 15),
                'proposed_month': random.randint(1, 12)
            })

        return pd.DataFrame(data)


if __name__ == '__main__':
    gen = CampaignDataGenerator()
    campaigns = gen.generate_historical_campaigns(100)
    print("Sample Historical Campaigns:")
    print(campaigns.head())
    print(f"\nGenerated {len(campaigns)} campaigns")
