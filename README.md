# trends2alpha
this project trains a random forest ML model to read and analyze past trends for various keywords, and utilizes that data to make purchasing decisions for stock tickers

## What It Does

- Collects Google Trends data for selected industry-related keywords  
- Maps keywords to relevant stock tickers (e.g. Apple → AAPL)  
- Combines keyword interest with daily stock price data  
- Trains a Random Forest classifier to predict next-day returns  
- Compares a Buy & Hold strategy with a keyword-driven strategy  
- Saves the results as strategy performance plots in the `plots/` folder  

---

## Concept

The idea is that search behavior may precede market moves.  
For example, increased search volume for “cancelled flights” may predict negative returns for airline stocks.  
This script attempts to capture that predictive power by combining trend data with stock return labels.

---

## Project Structure

```
alpha-trend-predictor/
├── trends_alpha.py         # Main script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── trend_cache/            # Cached keyword trend CSVs
└── plots/                  # Output PNG charts per ticker
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/alpha-trend-predictor.git
cd alpha-trend-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the script:

```bash
python trends_alpha.py
```

This will:
- Fetch and cache Google Trends data for each keyword
- Download historical stock prices
- Train a machine learning model per ticker
- Save strategy vs. market return plots in the `plots/` folder

---

## Output

Each output graph shows:

-  **Buy & Hold** – Cumulative return if you simply held the stock
-  **Model Strategy** – Return based on keyword-driven predictions

Saved as `.png` images inside the `plots/` directory.

---

## Notes on Limits

- Google Trends has **strict rate limits (429 errors)**  
- Cached results are reused from `trend_cache/` to avoid hitting the API  
- Trend data is **weekly**, stock data is **daily**  
  - The script uses backward `merge_asof` to align them appropriately  
- If there is not enough aligned data for a stock, it is skipped

---

## Future Ideas

- Add support for more ML models (XGBoost, Logistic Regression, etc.)
- Introduce industry sentiment scoring based on keyword groups
- Support real-time inference and alerting
- Add dashboard for visualizing live signals

---

## Requirements

Install with:

```bash
pip install -r requirements.txt
```

Example dependencies:

- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pytrends`

---

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute with attribution.

---

## Acknowledgments

- [PyTrends](https://github.com/GeneralMills/pytrends) – Google Trends API
- [yFinance](https://github.com/ranaroussi/yfinance) – Historical stock data
- [scikit-learn](https://scikit-learn.org) – ML model implementation
