# trends2alpha

This project trains a Random Forest ML model that combines **Google Trends data** with **technical indicators** to predict **monthly stock movements**, achieving significant outperformance during backtesting.

---

##  What It Does

-  Collects Google Trends data for stock-specific keywords (e.g., `"AAPL stock"`, `"Apple earnings"`)
-  Downloads 5 years of historical stock data and calculates technical indicators (RSI, SMA, volatility)
-  Identifies optimal time lags between search trends and price movements
-  Trains a Random Forest classifier to predict **monthly (not daily)** returns
-  Backtests a trend-informed strategy against buy-and-hold
-  Generates comprehensive performance visualizations and metrics

---

##  Concept

The hypothesis: **search behavior on Google can provide early signals for stock movements**.

> For example, spikes in `"NVDA stock"` searches might precede price increases by 1–2 weeks.

This project combines these **alternative data signals** with traditional **technical analysis** for enhanced predictions.

---

## 📊 Results

The model achieved strong performance during backtesting (2020–2024):

| Stock | AUC Score | Strategy Return | Market Return | Sharpe Ratio | Win Rate |
|-------|-----------|------------------|----------------|---------------|----------|
| AAPL  | 0.867     | 499.82%          | 240.48%        | 2.73          | 87.9%    |
| NVDA  | 0.956     | 10,461.16%       | 6,960.50%      | 2.65          | 82.9%    |
| AMZN  | 0.870     | 659.50%          | 484.77%        | 3.02          | 86.8%    |
| MSFT  | 0.867     | 187.49%          | 100.21%        | 2.36          | 84.6%    |
| GOOGL | 0.924     | 495.95%          | 279.85%        | 2.56          | 88.2%    |

>  **Important Disclaimers**:
> - Results reflect an exceptional bull market period (2020–2024)
> - Backtesting assumes perfect execution with **no transaction costs**
> - Past performance does not indicate future results
> - **Real-world returns would be significantly lower**

---

##  Key Features

### 📉 Technical Indicators
- **RSI**: Relative Strength Index for overbought/oversold conditions  
- **SMA**: 20 and 50-day moving averages  
- **Volatility**: 20-day rolling standard deviation  
- **Price Ratios**: e.g. Price relative to SMA

###  Google Trends Features
- **Trend Momentum**: Rate of change in search volume
- **Spike Detection**: Identifies unusual search activity (z-score > 2)
- **Lag Optimization**: Automatically finds optimal time delay (typically 1–2 weeks)

###  Machine Learning
- **Model**: Random Forest with 200 estimators
- **Validation**: 80/20 time-series split (no leakage)
- **Features**: ~20 combined trend + technical features per stock

---

## Installation

### 1. Clone the repository

  git clone https://github.com/ccloud24/alpha-trend-predictor.git
  cd alpha-trend-predictor

2. Create and activate a virtual environment
  python -m venv venv
  source venv/bin/activate       # On Windows: venv\Scripts\activate
3. Install dependencies
  pip install -r requirements.txt
Usage
  Run the main analysis:
    python trends_alpha.py
This will:

Fetch Google Trends data (or use cached results)

Download historical stock data via Yahoo Finance

Engineer technical + trend features

Train Random Forest models for each stock

Generate strategy performance plots and statistics

⏱ Expected runtime: 5–10 minutes (depending on cache status)

 Output
All results saved in the plots/ folder:

 Individual stock analysis: Cumulative return plots

 Summary dashboard: Comparative performance metrics

 CSV: model_performance_summary.csv with all model results

 Key Metrics Explained
Metric	Description
AUC Score:	Model's ability to distinguish up vs down months (>0.7 is good)
Sharpe Ratio:	Risk-adjusted return (>1 is good, >2 is excellent)
Win Rate:	Percent of months where model predicted direction correctly

 Methodology:
Data Collection: Weekly Google Trends + Daily Yahoo Finance prices

Features:

Technical indicators on price/volume

Google search trends: spikes, change rates, z-scores

Lag Optimization: Test -4 to +4 week lags to find best signal delay

Model Training: Random Forest Classifier with balanced weights

Backtesting: Monthly strategy rebalancing vs buy-and-hold benchmark


 Future Improvements

Support for crypto and commodity markets

Implement walk-forward validation

Introduce position sizing / risk management

Test on bear and sideways markets

Requirements:
yfinance>=0.2.18
pytrends>=4.9.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
ta>=0.10.2
📄 License
This project is licensed under the MIT License.
You're free to use, modify, and distribute with attribution.

🙏 Acknowledgments
pytrends – Unofficial Google Trends API

yfinance – Yahoo Finance data

scikit-learn – Machine Learning framework

ta – Technical analysis indicators for Python
