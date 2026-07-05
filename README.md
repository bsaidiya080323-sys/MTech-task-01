# 🎯 Charity Fundraising Campaign Optimizer

**Author:** Noor ul Huda  
**Registration Number:** Mtech-AI26092  
**Project Type:** Advanced | Tkinter + Scikit-learn | Campaign Optimization  
**Organization:** M-Tech Internship 2026 — AI/ML Week 1

---

## 📖 Overview

The **Charity Fundraising Campaign Optimizer** is an AI-powered desktop application that helps nonprofit organizations optimize their fundraising campaigns. Using machine learning models built with **scikit-learn** and an intuitive **Tkinter GUI**, the system predicts campaign performance, recommends optimal budget allocation, segments donors, and provides data-driven insights to maximize fundraising impact.

### Real Problem Solved

Charities often struggle with:
- **Inefficient budget allocation** across multiple campaigns
- **Poor timing** of fundraising efforts (missing peak seasons)
- **Lack of donor insights** leading to generic outreach
- **No predictive analytics** to forecast campaign success

This application addresses all these challenges using AI/ML algorithms.

---

## ✨ Features

### 🤖 Machine Learning Models
- **Amount Raised Predictor** (Gradient Boosting Regressor) — Predicts how much a campaign will raise
- **ROI Predictor** (Random Forest Regressor) — Forecasts return on investment
- **Success Classifier** (Random Forest Classifier) — Classifies campaigns as Highly Successful / Successful / Moderate / Underperformed
- **Donor Segmentation** (K-Means Clustering) — Groups donors into Champions, Loyal Supporters, Potential Donors, and At-Risk Donors

### ⚡ Optimization Engine
- **Budget Allocation Optimizer** — Distributes budget across campaigns using 3 strategies:
  - ROI Maximization
  - Risk-Adjusted Allocation
  - Diversified Portfolio
- **Campaign Selection** — Chooses the best combination of campaigns within budget constraints
- **Timing Optimizer** — Recommends optimal months based on seasonality patterns
- **Donor Strategy Generator** — Creates targeted strategies for each donor segment

### 📊 Interactive Dashboard
- Real-time KPI monitoring
- Tabbed interface: Dashboard, Data, ML Models, Optimizer, Donors, Reports
- Data export capabilities
- Comprehensive reporting

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download** the project files

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Pillow
- scipy

3. **Run the application:**
```bash
python main.py
```

---

## 📂 Project Structure

```
charity_fundraising_optimizer/
├── main.py              # Entry point - launches the application
├── gui.py               # Tkinter GUI with all tabs and controls
├── models.py            # ML models (scikit-learn implementations)
├── optimizer.py         # Campaign optimization algorithms
├── data_generator.py    # Realistic sample data generator
├── utils.py             # Visualization and helper functions
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── screenshots/         # Application screenshots
```

---

## 🧠 How AI Was Used

This project was built with AI assistance in the following ways:

1. **Learning & Research:** Used ChatGPT/Claude to understand fundraising optimization techniques, ML model selection, and scikit-learn best practices
2. **Code Structure:** AI helped design the modular architecture separating GUI, ML models, and optimization logic
3. **Algorithm Selection:** AI assisted in choosing appropriate algorithms (Gradient Boosting for regression, K-Means for clustering, Random Forest for classification)
4. **Debugging:** AI helped troubleshoot scikit-learn API issues and Tkinter widget configurations
5. **Documentation:** AI assisted in generating comprehensive documentation and README

**All core logic, design decisions, and understanding were developed independently.**

---

## 🖼️ Screenshots

### 1. Dashboard Overview
![Dashboard](screenshots/dashboard.png)
*Main dashboard showing KPIs and quick overview*

### 2. ML Model Training
![ML Models](screenshots/ml_models.png)
*Training machine learning models with performance metrics*

### 3. Campaign Optimization
![Optimizer](screenshots/optimizer.png)
*Running optimization with budget allocation results*

### 4. Donor Segmentation
![Donors](screenshots/donors.png)
*Analyzing donor segments and generating strategies*

---

## 📊 Algorithms Implemented

| Algorithm | Purpose | Library |
|-----------|---------|---------|
| Gradient Boosting Regressor | Amount prediction | scikit-learn |
| Random Forest Regressor | ROI prediction | scikit-learn |
| Random Forest Classifier | Success classification | scikit-learn |
| K-Means Clustering | Donor segmentation | scikit-learn |
| Greedy Optimization | Budget allocation | Custom (scipy) |
| Combinatorial Search | Campaign selection | Custom |
| Seasonality Analysis | Timing optimization | Custom |

---

## 🎯 Usage Workflow

1. **Generate Data** — Click "Generate Sample Data" to create campaign and donor datasets
2. **Train Models** — Click "Train ML Models" to build all 4 predictive models
3. **Run Optimization** — Set budget and parameters, then run optimization
4. **Analyze Donors** — Segment donors and view targeted strategies
5. **Generate Report** — Get a comprehensive optimization report

---

## 📈 Performance Metrics

Sample model performance on generated data:
- **Amount Predictor R²:** ~0.89
- **ROI Predictor R²:** ~0.84
- **Success Classifier Accuracy:** ~87%
- **Donor Segments:** 4 distinct clusters

---

## 🔮 Future Enhancements

- Integration with real charity CRM data
- A/B testing framework for campaigns
- Real-time donation tracking
- Multi-year forecasting
- Integration with email marketing APIs

---

## 📄 License

This project was created as part of the M-Tech Internship 2026 AI/ML program.

---

**Built with ❤️ for making the world a better place through data-driven philanthropy.**
