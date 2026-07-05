"""
Charity Fundraising Campaign Optimizer - Tkinter GUI
Main graphical user interface for the campaign optimization system.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import io
import os
import threading
import warnings
warnings.filterwarnings('ignore')

from data_generator import CampaignDataGenerator
from models import CampaignMLModels
from optimizer import CampaignOptimizer
from utils import (
    create_budget_allocation_chart, create_roi_comparison_chart,
    create_donor_segment_chart, create_performance_trend_chart,
    create_feature_importance_chart, fig_to_image
)

class CharityCampaignOptimizerApp:
    """Main application class for the Charity Fundraising Campaign Optimizer."""

    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Charity Fundraising Campaign Optimizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f6fa')

        # Initialize components
        self.data_gen = CampaignDataGenerator()
        self.ml_models = CampaignMLModels()
        self.optimizer = CampaignOptimizer()

        self.campaigns_df = None
        self.donors_df = None
        self.options_df = None
        self.optimized_df = None

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'), foreground='#2c3e50')
        self.style.configure('Subtitle.TLabel', font=('Helvetica', 12), foreground='#7f8c8d')
        self.style.configure('Card.TFrame', background='white')
        self.style.configure('Action.TButton', font=('Helvetica', 11, 'bold'), padding=10)

        self._create_widgets()
        self._generate_initial_data()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text="🎯 Charity Fundraising Campaign Optimizer", 
                 style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(header_frame, text="AI-Powered Campaign Optimization for Maximum Impact", 
                 style='Subtitle.TLabel').pack(anchor=tk.W)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self._create_dashboard_tab()
        self._create_data_tab()
        self._create_ml_models_tab()
        self._create_optimizer_tab()
        self._create_donors_tab()
        self._create_reports_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Generate data to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W, padding=(10, 5))
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _create_dashboard_tab(self):
        """Create the Dashboard tab."""
        self.dashboard_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.dashboard_frame, text="📊 Dashboard")

        # KPI Cards
        kpi_frame = ttk.Frame(self.dashboard_frame)
        kpi_frame.pack(fill=tk.X, pady=(0, 20))

        self.kpi_labels = {}
        kpi_data = [
            ('total_budget', '💰 Total Budget', '$0'),
            ('expected_raised', '📈 Expected Raised', '$0'),
            ('portfolio_roi', '🎯 Portfolio ROI', '0x'),
            ('num_campaigns', '📋 Campaigns', '0'),
            ('total_donors', '👥 Total Donors', '0'),
            ('model_status', '🤖 ML Status', 'Not Trained')
        ]

        for i, (key, title, value) in enumerate(kpi_data):
            card = tk.Frame(kpi_frame, bg='white', bd=2, relief=tk.GROOVE, padx=15, pady=15)
            card.grid(row=0, column=i, padx=10, pady=5, sticky='nsew')
            kpi_frame.grid_columnconfigure(i, weight=1)

            tk.Label(card, text=title, font=('Helvetica', 11, 'bold'), 
                    bg='white', fg='#2c3e50').pack(anchor=tk.W)
            lbl = tk.Label(card, text=value, font=('Helvetica', 18, 'bold'), 
                          bg='white', fg='#667eea')
            lbl.pack(anchor=tk.W, pady=(5, 0))
            self.kpi_labels[key] = lbl

        # Action buttons
        btn_frame = ttk.Frame(self.dashboard_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="🔄 Generate Sample Data", 
                  command=self._generate_initial_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🧠 Train ML Models", 
                  command=self._train_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⚡ Run Optimization", 
                  command=self._run_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📄 Generate Report", 
                  command=self._generate_report).pack(side=tk.LEFT, padx=5)

        # Quick stats text area
        stats_frame = ttk.LabelFrame(self.dashboard_frame, text="Quick Overview", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.dashboard_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, 
                                                       font=('Consolas', 10), height=15)
        self.dashboard_text.pack(fill=tk.BOTH, expand=True)
        self.dashboard_text.insert(tk.END, "Welcome to the Charity Fundraising Campaign Optimizer!\n\n")
        self.dashboard_text.insert(tk.END, "Steps to get started:\n")
        self.dashboard_text.insert(tk.END, "1. Click 'Generate Sample Data' to create campaign and donor data\n")
        self.dashboard_text.insert(tk.END, "2. Click 'Train ML Models' to build predictive models\n")
        self.dashboard_text.insert(tk.END, "3. Click 'Run Optimization' to get optimal budget allocation\n")
        self.dashboard_text.insert(tk.END, "4. Explore other tabs for detailed analysis\n")
        self.dashboard_text.config(state=tk.DISABLED)

    def _create_data_tab(self):
        """Create the Data tab."""
        self.data_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.data_frame, text="📁 Data")

        # Controls
        ctrl_frame = ttk.Frame(self.data_frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl_frame, text="Data Source:").pack(side=tk.LEFT, padx=(0, 10))
        self.data_source_var = tk.StringVar(value="campaigns")
        ttk.Radiobutton(ctrl_frame, text="Campaigns", variable=self.data_source_var, 
                       value="campaigns", command=self._refresh_data_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ctrl_frame, text="Donors", variable=self.data_source_var, 
                       value="donors", command=self._refresh_data_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ctrl_frame, text="Options", variable=self.data_source_var, 
                       value="options", command=self._refresh_data_view).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl_frame, text="💾 Export to CSV", 
                  command=self._export_data).pack(side=tk.RIGHT, padx=5)

        # Treeview for data display
        tree_frame = ttk.Frame(self.data_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.data_tree = ttk.Treeview(tree_frame)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=scrollbar.set)

    def _create_ml_models_tab(self):
        """Create the ML Models tab."""
        self.ml_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.ml_frame, text="🧠 ML Models")

        # Model training controls
        train_frame = ttk.LabelFrame(self.ml_frame, text="Model Training", padding="15")
        train_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(train_frame, text="Training Dataset Size:").pack(side=tk.LEFT, padx=(0, 10))
        self.train_size_var = tk.IntVar(value=500)
        ttk.Spinbox(train_frame, from_=100, to=2000, textvariable=self.train_size_var, 
                   width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_frame, text="🚀 Train All Models", 
                  command=self._train_models).pack(side=tk.LEFT, padx=20)

        # Model results
        results_frame = ttk.LabelFrame(self.ml_frame, text="Model Performance", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.ml_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD,
                                                          font=('Consolas', 10), height=20)
        self.ml_results_text.pack(fill=tk.BOTH, expand=True)
        self.ml_results_text.insert(tk.END, "ML Models not yet trained. Click 'Train All Models' to begin.\n")
        self.ml_results_text.config(state=tk.DISABLED)

    def _create_optimizer_tab(self):
        """Create the Optimizer tab."""
        self.opt_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.opt_frame, text="⚡ Optimizer")

        # Optimization controls
        ctrl_frame = ttk.LabelFrame(self.opt_frame, text="Optimization Settings", padding="15")
        ctrl_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(ctrl_frame, text="Total Budget ($):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.budget_var = tk.DoubleVar(value=50000)
        ttk.Entry(ctrl_frame, textvariable=self.budget_var, width=15).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="Optimization Method:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.opt_method_var = tk.StringVar(value="roi_maximization")
        ttk.Combobox(ctrl_frame, textvariable=self.opt_method_var, 
                      values=["roi_maximization", "risk_adjusted", "diversified"],
                      width=20, state="readonly").grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="Max Campaigns:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_campaigns_var = tk.IntVar(value=5)
        ttk.Spinbox(ctrl_frame, from_=1, to=10, textvariable=self.max_campaigns_var, 
                   width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="Min ROI Threshold:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.min_roi_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(ctrl_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.min_roi_var, 
                   width=10).grid(row=1, column=3, padx=5, pady=5)

        ttk.Button(ctrl_frame, text="▶ Run Optimization", 
                  command=self._run_optimization).grid(row=2, column=0, columnspan=4, pady=15)

        # Results
        results_frame = ttk.LabelFrame(self.opt_frame, text="Optimization Results", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.opt_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD,
                                                           font=('Consolas', 10), height=20)
        self.opt_results_text.pack(fill=tk.BOTH, expand=True)
        self.opt_results_text.insert(tk.END, "Run optimization to see results here.\n")
        self.opt_results_text.config(state=tk.DISABLED)

    def _create_donors_tab(self):
        """Create the Donors tab."""
        self.donors_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.donors_frame, text="👥 Donors")

        ctrl_frame = ttk.Frame(self.donors_frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(ctrl_frame, text="🔍 Analyze Segments", 
                  command=self._analyze_donors).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="📊 Generate Strategies", 
                  command=self._generate_strategies).pack(side=tk.LEFT, padx=5)

        self.donors_text = scrolledtext.ScrolledText(self.donors_frame, wrap=tk.WORD,
                                                      font=('Consolas', 10))
        self.donors_text.pack(fill=tk.BOTH, expand=True)
        self.donors_text.insert(tk.END, "Click 'Analyze Segments' to perform donor segmentation.\n")
        self.donors_text.config(state=tk.DISABLED)

    def _create_reports_tab(self):
        """Create the Reports tab."""
        self.reports_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.reports_frame, text="📄 Reports")

        ctrl_frame = ttk.Frame(self.reports_frame)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(ctrl_frame, text="📋 Full Optimization Report", 
                  command=self._generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="💾 Save Report to File", 
                  command=self._save_report).pack(side=tk.LEFT, padx=5)

        self.reports_text = scrolledtext.ScrolledText(self.reports_frame, wrap=tk.WORD,
                                                      font=('Consolas', 10))
        self.reports_text.pack(fill=tk.BOTH, expand=True)
        self.reports_text.insert(tk.END, "Generate a report to see comprehensive optimization analysis.\n")
        self.reports_text.config(state=tk.DISABLED)

    def _generate_initial_data(self):
        """Generate initial sample data."""
        self.status_var.set("Generating sample data...")
        self.root.update()

        try:
            self.campaigns_df = self.data_gen.generate_historical_campaigns(500)
            self.donors_df = self.data_gen.generate_donor_profiles(1000)
            self.options_df = self.data_gen.generate_current_campaign_options(20)

            self._update_kpi()
            self._refresh_data_view()
            self._update_dashboard_text()

            self.status_var.set(f"Data generated: {len(self.campaigns_df)} campaigns, {len(self.donors_df)} donors")
            messagebox.showinfo("Success", f"Generated {len(self.campaigns_df)} campaigns and {len(self.donors_df)} donor profiles!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
            self.status_var.set("Error generating data")

    def _update_kpi(self):
        """Update KPI labels on dashboard."""
        if self.campaigns_df is not None:
            total_budget = self.campaigns_df['budget'].sum()
            avg_roi = self.campaigns_df['roi'].mean()
            num_campaigns = len(self.campaigns_df)

            self.kpi_labels['total_budget'].config(text=f"${total_budget:,.0f}")
            self.kpi_labels['portfolio_roi'].config(text=f"{avg_roi:.2f}x")
            self.kpi_labels['num_campaigns'].config(text=str(num_campaigns))

        if self.donors_df is not None:
            self.kpi_labels['total_donors'].config(text=str(len(self.donors_df)))

    def _update_dashboard_text(self):
        """Update dashboard text area."""
        self.dashboard_text.config(state=tk.NORMAL)
        self.dashboard_text.delete(1.0, tk.END)

        if self.campaigns_df is not None:
            self.dashboard_text.insert(tk.END, "📊 Campaign Data Summary\n")
            self.dashboard_text.insert(tk.END, "=" * 50 + "\n")
            self.dashboard_text.insert(tk.END, f"Total Campaigns: {len(self.campaigns_df)}\n")
            self.dashboard_text.insert(tk.END, f"Total Budget: ${self.campaigns_df['budget'].sum():,.2f}\n")
            self.dashboard_text.insert(tk.END, f"Total Raised: ${self.campaigns_df['amount_raised'].sum():,.2f}\n")
            self.dashboard_text.insert(tk.END, f"Average ROI: {self.campaigns_df['roi'].mean():.2f}x\n")
            self.dashboard_text.insert(tk.END, f"Success Rate: {(self.campaigns_df['success_label'] != 'Underperformed').mean():.1%}\n\n")

            self.dashboard_text.insert(tk.END, "Top Performing Campaign Types:\n")
            type_perf = self.campaigns_df.groupby('campaign_type')['roi'].mean().sort_values(ascending=False).head(5)
            for ctype, roi in type_perf.items():
                self.dashboard_text.insert(tk.END, f"  • {ctype}: {roi:.2f}x ROI\n")

        if self.donors_df is not None:
            self.dashboard_text.insert(tk.END, "\n👥 Donor Data Summary\n")
            self.dashboard_text.insert(tk.END, "=" * 50 + "\n")
            self.dashboard_text.insert(tk.END, f"Total Donors: {len(self.donors_df)}\n")
            self.dashboard_text.insert(tk.END, f"Avg Donation: ${self.donors_df['avg_monetary'].mean():.2f}\n")
            self.dashboard_text.insert(tk.END, f"High Churn Risk: {(self.donors_df['churn_risk'] == 'High').sum()} donors\n")

        self.dashboard_text.config(state=tk.DISABLED)

    def _refresh_data_view(self):
        """Refresh the data treeview."""
        # Clear existing
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        source = self.data_source_var.get()

        if source == "campaigns" and self.campaigns_df is not None:
            df = self.campaigns_df.head(100)
            columns = ['campaign_id', 'campaign_type', 'channel', 'budget', 'roi', 'success_label']
        elif source == "donors" and self.donors_df is not None:
            df = self.donors_df.head(100)
            columns = ['donor_id', 'age', 'avg_monetary', 'rfm_score', 'churn_risk']
        elif source == "options" and self.options_df is not None:
            df = self.options_df.head(100)
            columns = list(df.columns)
        else:
            return

        # Configure columns
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'

        for col in columns:
            self.data_tree.heading(col, text=col.replace('_', ' ').title())
            self.data_tree.column(col, width=100)

        # Insert data
        for _, row in df.iterrows():
            values = [row.get(col, '') for col in columns]
            self.data_tree.insert('', tk.END, values=values)

    def _train_models(self):
        """Train all ML models."""
        if self.campaigns_df is None or self.donors_df is None:
            messagebox.showwarning("Warning", "Please generate data first!")
            return

        self.status_var.set("Training ML models...")
        self.root.update()

        try:
            self.ml_results_text.config(state=tk.NORMAL)
            self.ml_results_text.delete(1.0, tk.END)
            self.ml_results_text.insert(tk.END, "🧠 Training Machine Learning Models...\n")
            self.ml_results_text.insert(tk.END, "=" * 60 + "\n\n")
            self.ml_results_text.update()

            # Train amount predictor
            self.ml_results_text.insert(tk.END, "1️⃣ Training Amount Raised Predictor\n")
            self.ml_results_text.insert(tk.END, "-" * 40 + "\n")
            amount_results = self.ml_models.train_amount_predictor(self.campaigns_df)
            self.ml_results_text.insert(tk.END, f"   RMSE: ${amount_results['rmse']:,.2f}\n")
            self.ml_results_text.insert(tk.END, f"   R² Score: {amount_results['r2']:.4f}\n\n")
            self.ml_results_text.update()

            # Train ROI predictor
            self.ml_results_text.insert(tk.END, "2️⃣ Training ROI Predictor\n")
            self.ml_results_text.insert(tk.END, "-" * 40 + "\n")
            roi_results = self.ml_models.train_roi_predictor(self.campaigns_df)
            self.ml_results_text.insert(tk.END, f"   RMSE: {roi_results['rmse']:.4f}\n")
            self.ml_results_text.insert(tk.END, f"   R² Score: {roi_results['r2']:.4f}\n\n")

            # Feature importance
            self.ml_results_text.insert(tk.END, "   Feature Importance:\n")
            for _, row in roi_results['feature_importance'].head(5).iterrows():
                self.ml_results_text.insert(tk.END, f"     • {row['feature']}: {row['importance']:.4f}\n")
            self.ml_results_text.insert(tk.END, "\n")
            self.ml_results_text.update()

            # Train success classifier
            self.ml_results_text.insert(tk.END, "3️⃣ Training Success Classifier\n")
            self.ml_results_text.insert(tk.END, "-" * 40 + "\n")
            success_results = self.ml_models.train_success_classifier(self.campaigns_df)
            self.ml_results_text.insert(tk.END, f"   Accuracy: {success_results['accuracy']:.4f}\n\n")
            self.ml_results_text.update()

            # Train donor segmentation
            self.ml_results_text.insert(tk.END, "4️⃣ Training Donor Segmentation (K-Means)\n")
            self.ml_results_text.insert(tk.END, "-" * 40 + "\n")
            seg_results = self.ml_models.train_donor_segmentation(self.donors_df)
            self.ml_results_text.insert(tk.END, f"   Segments found: {seg_results['analysis'].shape[0]}\n")
            for seg_name in seg_results['segments']['segment_name'].unique():
                count = (seg_results['segments']['segment_name'] == seg_name).sum()
                self.ml_results_text.insert(tk.END, f"     • {seg_name}: {count} donors\n")

            self.donors_df = seg_results['segments']

            self.ml_results_text.insert(tk.END, "\n✅ All models trained successfully!\n")
            self.ml_results_text.config(state=tk.DISABLED)

            self.kpi_labels['model_status'].config(text="Trained ✓")
            self.status_var.set("ML models trained successfully")

            messagebox.showinfo("Success", "All ML models trained successfully!")

        except Exception as e:
            self.ml_results_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
            self.ml_results_text.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Training failed")

    def _run_optimization(self):
        """Run campaign optimization."""
        if self.ml_models is None or not self.ml_models.is_trained:
            messagebox.showwarning("Warning", "Please train ML models first!")
            return

        if self.options_df is None:
            messagebox.showwarning("Warning", "Please generate data first!")
            return

        self.status_var.set("Running optimization...")
        self.root.update()

        try:
            # Predict performance for all options
            options = self.options_df.copy()
            predictions = []

            for _, row in options.iterrows():
                pred = self.ml_models.predict_campaign_performance(row.to_frame().T)
                predictions.append(pred)

            # Add predictions to options
            options['predicted_roi'] = [p.get('roi', 0) for p in predictions]
            options['predicted_amount'] = [p.get('amount_raised', 0) for p in predictions]
            options['predicted_success'] = [p.get('success_label', 'Unknown') for p in predictions]

            # Get optimization parameters
            total_budget = self.budget_var.get()
            method = self.opt_method_var.get()
            max_campaigns = self.max_campaigns_var.get()
            min_roi = self.min_roi_var.get()

            # Run optimization
            optimized = self.optimizer.optimize_budget_allocation(options, total_budget, method=method)

            # Select best campaigns
            selected = self.optimizer.select_optimal_campaigns(options, total_budget, 
                                                             max_campaigns=max_campaigns, 
                                                             min_roi=min_roi)

            # Optimize timing
            timing = self.optimizer.optimize_campaign_timing(options)

            # Calculate portfolio metrics
            metrics = self.optimizer.calculate_portfolio_metrics(optimized)

            # Store results
            self.optimized_df = optimized

            # Display results
            self.opt_results_text.config(state=tk.NORMAL)
            self.opt_results_text.delete(1.0, tk.END)

            self.opt_results_text.insert(tk.END, "⚡ CAMPAIGN OPTIMIZATION RESULTS\n")
            self.opt_results_text.insert(tk.END, "=" * 60 + "\n\n")

            self.opt_results_text.insert(tk.END, f"💰 Total Budget: ${total_budget:,.2f}\n")
            self.opt_results_text.insert(tk.END, f"📊 Optimization Method: {method.replace('_', ' ').title()}\n")
            self.opt_results_text.insert(tk.END, f"🎯 Portfolio ROI: {metrics['portfolio_roi']:.2f}x\n")
            self.opt_results_text.insert(tk.END, f"📈 Expected Return: ${metrics['expected_return']:,.2f}\n")
            self.opt_results_text.insert(tk.END, f"💵 Expected Amount Raised: ${metrics['expected_amount_raised']:,.2f}\n")
            self.opt_results_text.insert(tk.END, f"📉 ROI Volatility: {metrics['roi_volatility']:.4f}\n")
            self.opt_results_text.insert(tk.END, f"⚖️ Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
            self.opt_results_text.insert(tk.END, f"🔄 Diversification: {metrics['diversification_score']:.2%}\n\n")

            self.opt_results_text.insert(tk.END, "📋 OPTIMIZED BUDGET ALLOCATION\n")
            self.opt_results_text.insert(tk.END, "-" * 60 + "\n")

            for _, row in optimized.iterrows():
                self.opt_results_text.insert(tk.END, 
                    f"\n{row['campaign_type']}\n"
                    f"  Budget: ${row['optimized_budget']:,.2f} | "
                    f"ROI: {row['predicted_roi']:.2f}x | "
                    f"Expected: ${row['expected_return']:,.2f}\n"
                )

            self.opt_results_text.insert(tk.END, "\n📅 TIMING RECOMMENDATIONS\n")
            self.opt_results_text.insert(tk.END, "-" * 60 + "\n")
            for _, row in timing.iterrows():
                self.opt_results_text.insert(tk.END, 
                    f"  {row['campaign_type']}: Month {row['recommended_month']} "
                    f"(Boost: {row['seasonality_boost']:.2f}x)\n"
                )

            self.opt_results_text.config(state=tk.DISABLED)

            # Update KPIs
            self.kpi_labels['expected_raised'].config(text=f"${metrics['expected_amount_raised']:,.0f}")
            self.kpi_labels['portfolio_roi'].config(text=f"{metrics['portfolio_roi']:.2f}x")

            self.status_var.set("Optimization complete")
            messagebox.showinfo("Success", "Optimization complete! Check the results.")

        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")
            self.status_var.set("Optimization failed")

    def _analyze_donors(self):
        """Analyze donor segments."""
        if self.donors_df is None:
            messagebox.showwarning("Warning", "Please generate data first!")
            return

        if 'segment_name' not in self.donors_df.columns:
            messagebox.showwarning("Warning", "Please train ML models first to segment donors!")
            return

        self.donors_text.config(state=tk.NORMAL)
        self.donors_text.delete(1.0, tk.END)

        self.donors_text.insert(tk.END, "👥 DONOR SEGMENTATION ANALYSIS\n")
        self.donors_text.insert(tk.END, "=" * 60 + "\n\n")

        for segment in self.donors_df['segment_name'].unique():
            seg_data = self.donors_df[self.donors_df['segment_name'] == segment]

            self.donors_text.insert(tk.END, f"📌 {segment}\n")
            self.donors_text.insert(tk.END, f"   Size: {len(seg_data)} donors\n")
            self.donors_text.insert(tk.END, f"   Avg Age: {seg_data['age'].mean():.1f}\n")
            self.donors_text.insert(tk.END, f"   Avg Income: ${seg_data['income'].mean():,.2f}\n")
            self.donors_text.insert(tk.END, f"   Avg Donation: ${seg_data['avg_monetary'].mean():.2f}\n")
            self.donors_text.insert(tk.END, f"   Total Given: ${seg_data['total_donations'].sum():,.2f}\n")
            self.donors_text.insert(tk.END, f"   Avg RFM Score: {seg_data['rfm_score'].mean():.2f}\n")
            self.donors_text.insert(tk.END, f"   Churn Risk Distribution:\n")
            for risk, count in seg_data['churn_risk'].value_counts().items():
                self.donors_text.insert(tk.END, f"      {risk}: {count}\n")
            self.donors_text.insert(tk.END, "\n")

        self.donors_text.config(state=tk.DISABLED)

    def _generate_strategies(self):
        """Generate donor segment strategies."""
        if self.donors_df is None or 'segment_name' not in self.donors_df.columns:
            messagebox.showwarning("Warning", "Please train ML models first!")
            return

        strategies = self.optimizer.donor_segment_strategy(self.donors_df)

        self.donors_text.config(state=tk.NORMAL)
        self.donors_text.delete(1.0, tk.END)

        self.donors_text.insert(tk.END, "🎯 DONOR SEGMENT STRATEGIES\n")
        self.donors_text.insert(tk.END, "=" * 60 + "\n\n")

        for _, row in strategies.iterrows():
            self.donors_text.insert(tk.END, f"📌 {row['segment']}\n")
            self.donors_text.insert(tk.END, f"   Size: {row['size']} donors\n")
            self.donors_text.insert(tk.END, f"   Avg Donation: ${row['avg_donation']:.2f}\n")
            self.donors_text.insert(tk.END, f"   Approach: {row['recommended_approach']}\n")
            self.donors_text.insert(tk.END, f"   Campaign Types: {', '.join(row['campaign_types'])}\n")
            self.donors_text.insert(tk.END, f"   Frequency: {row['frequency']}\n")
            self.donors_text.insert(tk.END, f"   Expected Response: {row['expected_response_rate']:.1%}\n")
            self.donors_text.insert(tk.END, f"   Priority: {row['priority']}\n\n")

        self.donors_text.config(state=tk.DISABLED)

    def _generate_report(self):
        """Generate comprehensive optimization report."""
        if self.optimized_df is None:
            messagebox.showwarning("Warning", "Please run optimization first!")
            return

        try:
            metrics = self.optimizer.calculate_portfolio_metrics(self.optimized_df)

            timing = None
            if self.options_df is not None:
                timing = self.optimizer.optimize_campaign_timing(self.options_df)

            donor_strategies = None
            if self.donors_df is not None and 'segment_name' in self.donors_df.columns:
                donor_strategies = self.optimizer.donor_segment_strategy(self.donors_df)

            report = self.optimizer.generate_optimization_report(
                self.optimized_df, metrics, timing, donor_strategies
            )

            self.reports_text.config(state=tk.NORMAL)
            self.reports_text.delete(1.0, tk.END)
            self.reports_text.insert(tk.END, report)
            self.reports_text.config(state=tk.DISABLED)

            self.status_var.set("Report generated")

        except Exception as e:
            messagebox.showerror("Error", f"Report generation failed: {str(e)}")

    def _save_report(self):
        """Save report to file."""
        if self.reports_text.get(1.0, tk.END).strip() == "":
            messagebox.showwarning("Warning", "No report to save. Generate one first!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            with open(filepath, 'w') as f:
                f.write(self.reports_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Report saved to {filepath}")

    def _export_data(self):
        """Export current data view to CSV."""
        source = self.data_source_var.get()

        if source == "campaigns" and self.campaigns_df is not None:
            df = self.campaigns_df
            default_name = "campaigns.csv"
        elif source == "donors" and self.donors_df is not None:
            df = self.donors_df
            default_name = "donors.csv"
        elif source == "options" and self.options_df is not None:
            df = self.options_df
            default_name = "options.csv"
        else:
            messagebox.showwarning("Warning", "No data to export!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            df.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Data exported to {filepath}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = CharityCampaignOptimizerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
