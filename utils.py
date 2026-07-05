"""
Charity Fundraising Campaign Optimizer - Utility Functions
Helper functions for data processing, visualization, and reporting.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import base64

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_budget_allocation_chart(campaigns_df, figsize=(10, 6)):
    """Create a pie chart showing budget allocation."""
    fig, ax = plt.subplots(figsize=figsize)

    labels = campaigns_df['campaign_type'].values
    sizes = campaigns_df['optimized_budget'].values
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Optimized Budget Allocation', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_roi_comparison_chart(campaigns_df, figsize=(10, 6)):
    """Create a bar chart comparing predicted ROI across campaigns."""
    fig, ax = plt.subplots(figsize=figsize)

    campaigns_sorted = campaigns_df.sort_values('predicted_roi', ascending=True)

    colors = ['#e74c3c' if roi < 0.5 else '#f39c12' if roi < 1.0 else '#27ae60' 
              for roi in campaigns_sorted['predicted_roi']]

    bars = ax.barh(campaigns_sorted['campaign_type'], campaigns_sorted['predicted_roi'], color=colors)

    ax.set_xlabel('Predicted ROI', fontsize=12)
    ax.set_title('Campaign ROI Comparison', fontsize=14, fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even (ROI=1.0)')
    ax.legend()

    # Add value labels
    for bar, roi in zip(bars, campaigns_sorted['predicted_roi']):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{roi:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    return fig

def create_donor_segment_chart(segments_df, figsize=(10, 6)):
    """Create visualization of donor segments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Segment sizes
    segment_counts = segments_df['segment_name'].value_counts()
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Donor Segment Distribution', fontsize=12, fontweight='bold')

    # Average donation by segment
    avg_donation = segments_df.groupby('segment_name')['avg_monetary'].mean().sort_values(ascending=True)
    ax2.barh(avg_donation.index, avg_donation.values, color=colors)
    ax2.set_xlabel('Average Donation ($)', fontsize=11)
    ax2.set_title('Avg Donation by Segment', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def create_performance_trend_chart(historical_df, figsize=(10, 6)):
    """Create trend chart of historical campaign performance."""
    fig, ax = plt.subplots(figsize=figsize)

    monthly = historical_df.groupby('month').agg({
        'amount_raised': 'mean',
        'roi': 'mean',
        'num_donors': 'mean'
    }).reset_index()

    ax2 = ax.twinx()

    line1 = ax.plot(monthly['month'], monthly['amount_raised'], 'b-o', 
                    label='Avg Amount Raised', linewidth=2, markersize=8)
    line2 = ax2.plot(monthly['month'], monthly['roi'], 'r-s', 
                     label='Avg ROI', linewidth=2, markersize=8)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Amount Raised ($)', fontsize=12, color='b')
    ax2.set_ylabel('ROI', fontsize=12, color='r')
    ax.set_title('Historical Campaign Performance by Month', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    return fig

def create_feature_importance_chart(importance_df, figsize=(10, 6)):
    """Create feature importance chart."""
    fig, ax = plt.subplots(figsize=figsize)

    importance_sorted = importance_df.sort_values('importance', ascending=True)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_sorted)))
    ax.barh(importance_sorted['feature'], importance_sorted['importance'], color=colors)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance for Campaign Success', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def fig_to_image(fig, dpi=100):
    """Convert matplotlib figure to PhotoImage for Tkinter."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = buf.getvalue()
    buf.close()
    plt.close(fig)
    return img

def generate_sample_report_data():
    """Generate sample data for demonstration."""
    campaigns = pd.DataFrame({
        'campaign_type': ['Email Campaign', 'Social Media', 'Event Fundraising', 
                         'Direct Mail', 'Corporate Partnership', 'Peer-to-Peer'],
        'optimized_budget': [8000, 5000, 20000, 7000, 15000, 5000],
        'predicted_roi': [2.5, 1.8, 3.2, 1.5, 2.8, 2.0],
        'expected_return': [20000, 9000, 64000, 10500, 42000, 10000],
        'expected_amount_raised': [28000, 14000, 84000, 17500, 57000, 15000]
    })

    segments = pd.DataFrame({
        'segment_name': ['Champions', 'Loyal Supporters', 'Potential Donors', 'At-Risk Donors'] * 250,
        'avg_monetary': [500, 200, 75, 50] * 250
    })

    historical = pd.DataFrame({
        'month': np.random.randint(1, 13, 500),
        'amount_raised': np.random.lognormal(8, 1, 500),
        'roi': np.random.normal(1.5, 0.8, 500),
        'num_donors': np.random.poisson(100, 500)
    })

    importance = pd.DataFrame({
        'feature': ['engagement_rate', 'budget', 'campaign_type', 'duration_days', 
                   'outreach_count', 'month', 'channel', 'target_audience', 'cause'],
        'importance': [0.28, 0.22, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02, 0.00]
    })

    return campaigns, segments, historical, importance


if __name__ == '__main__':
    campaigns, segments, historical, importance = generate_sample_report_data()

    fig1 = create_budget_allocation_chart(campaigns)
    fig1.savefig('/mnt/agents/output/charity_fundraising_optimizer/screenshots/budget_allocation.png')

    fig2 = create_roi_comparison_chart(campaigns)
    fig2.savefig('/mnt/agents/output/charity_fundraising_optimizer/screenshots/roi_comparison.png')

    fig3 = create_donor_segment_chart(segments)
    fig3.savefig('/mnt/agents/output/charity_fundraising_optimizer/screenshots/donor_segments.png')

    print("Sample charts created successfully!")
