import os
import time
import warnings
import yfinance as yf
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Improved keywords - focusing on actual trading sentiment
industry_keywords = {
    'tech_stocks': {
        'keywords': ['AAPL stock', 'Apple earnings', 'iPhone sales', 
                    'NVDA stock', 'Nvidia earnings', 'AI chip demand'],
        'tickers': ['AAPL', 'NVDA']
    },
    'retail': {
        'keywords': ['AMZN stock', 'Amazon earnings', 'Prime Day sales'],
        'tickers': ['AMZN']
    },
    'tech_giants': {
        'keywords': ['MSFT stock', 'Microsoft earnings', 'Azure growth',
                    'GOOGL stock', 'Google earnings', 'YouTube revenue'],
        'tickers': ['MSFT', 'GOOGL']
    }
}

# Mapping to help with keyword-ticker matching
ticker_mapping = {
    'AAPL': ['AAPL', 'Apple'],
    'NVDA': ['NVDA', 'Nvidia'],
    'AMZN': ['AMZN', 'Amazon'],
    'MSFT': ['MSFT', 'Microsoft'],
    'GOOGL': ['GOOGL', 'Google']
}

start_date = '2020-01-01'
end_date = '2024-12-31'
trend_cache_dir = "trend_cache"
plot_dir = "plots"

os.makedirs(trend_cache_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

pytrends = TrendReq(hl='en-US', tz=-300)

def get_cached_or_fetch_keyword(keyword, timeframe="today 5-y"):
    """Fetch Google Trends data with caching"""
    cache_path = os.path.join(trend_cache_dir, f"{keyword.replace(' ', '_')}.csv")
    
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return process_trend_data(df, keyword)
    
    for attempt in range(3):
        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            trend_df = pytrends.interest_over_time()
            if trend_df.empty:
                return None
            
            # Clean and save
            trend_df = trend_df[~trend_df['isPartial']]
            trend_df.to_csv(cache_path)
            time.sleep(5)  # Be respectful to Google's servers
            return process_trend_data(trend_df, keyword)
            
        except Exception as e:
            print(f"❌ Error fetching {keyword} (attempt {attempt + 1}): {e}")
            time.sleep(10)
    return None

def process_trend_data(trend_df, keyword):
    """Extract useful features from Google Trends data"""
    df = pd.DataFrame()
    
    # Handle if keyword is not in columns (from cached data)
    if keyword in trend_df.columns:
        df['trend'] = trend_df[keyword]
    else:
        # Assume first non-isPartial column is the trend data
        trend_cols = [col for col in trend_df.columns if col != 'isPartial']
        if trend_cols:
            df['trend'] = trend_df[trend_cols[0]]
    
    # Enhanced features that will actually be useful
    df['trend_ma4'] = df['trend'].rolling(4).mean()  # 4-week MA
    df['trend_change_4w'] = df['trend'].pct_change(4)  # 4-week change
    df['trend_momentum'] = df['trend'] - df['trend_ma4']  # Momentum
    df['trend_acceleration'] = df['trend_momentum'].diff()  # Acceleration
    
    # Spike detection (significant deviations)
    rolling_mean = df['trend'].rolling(12).mean()
    rolling_std = df['trend'].rolling(12).std()
    df['trend_zscore'] = (df['trend'] - rolling_mean) / (rolling_std + 0.0001)
    df['trend_spike'] = (np.abs(df['trend_zscore']) > 2).astype(int)
    
    # Trend strength
    df['trend_strength'] = df['trend'] / (df['trend_ma4'] + 0.0001)
    
    return df.dropna()

def add_technical_features(df):
    """Add simple but effective technical indicators"""
    close = df['Close']
    volume = df['Volume']
    
    # Returns - including monthly
    df['returns'] = close.pct_change()
    df['returns_20d'] = close.pct_change(20)  # Monthly return
    
    # Moving averages
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['price_to_sma20'] = close / df['sma_20']
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.0001)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume indicator
    df['volume_ratio'] = volume / volume.rolling(20).mean()
    
    return df

# Load trend data
print("🔍 Fetching Google Trends data...")
keyword_trend_data = {}
for industry, config in industry_keywords.items():
    for keyword in config['keywords']:
        print(f"🔍 Fetching trend for {keyword}")
        trend_df = get_cached_or_fetch_keyword(keyword)
        if trend_df is not None:
            keyword_trend_data[keyword] = trend_df
            print(f"   ✓ Got {len(trend_df)} rows of trend data")

# Load and process stock data
print("\n📥 Downloading stock data...")
ticker_datasets = {}
for industry, config in industry_keywords.items():
    for ticker in config['tickers']:
        if ticker not in ticker_datasets:
            print(f"📥 Downloading {ticker}")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                # Handle multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                print(f"📊 Columns for {ticker}: {list(df.columns)}")
                print(f"📏 Shape of {ticker} data: {df.shape}")
                
                # Add technical features
                df = add_technical_features(df)
                initial_len = len(df)
                df = df.dropna()
                print(f"   After technical features: {len(df)} rows (dropped {initial_len - len(df)} NaN rows)")
                
                # Create weekly version for better trend alignment
                df_weekly = df.resample('W').agg({
                    'Close': 'last',
                    'returns': 'sum',
                    'returns_20d': 'last',
                    'sma_20': 'last',
                    'sma_50': 'last',
                    'price_to_sma20': 'last',
                    'rsi': 'last',
                    'volatility': 'mean',
                    'volume_ratio': 'mean'
                }).dropna()
                
                ticker_datasets[ticker] = df_weekly
                print(f"   Weekly data: {len(df_weekly)} rows")
                
            except Exception as e:
                print(f"⚠️ Error processing {ticker}: {e}")
                continue
        
        # Count successful merges
        merge_count = 0
        
        # Merge trend data for keywords related to this ticker
        for keyword in keyword_trend_data.keys():
            # Check if keyword is relevant to this ticker
            keyword_relevant = False
            if ticker in ticker_mapping:
                for term in ticker_mapping[ticker]:
                    if term.lower() in keyword.lower():
                        keyword_relevant = True
                        break
            
            if keyword_relevant:
                trend_df = keyword_trend_data[keyword]
                suffix = keyword.lower().replace(' ', '_')
                
                print(f"   Merging {keyword} data for {ticker}")
                
                # Ensure trend data is also weekly
                trend_df_weekly = trend_df.resample('W').last().dropna()
                
                # Rename columns to avoid conflicts
                trend_cols_to_merge = ['trend_ma4', 'trend_change_4w', 'trend_momentum', 
                                     'trend_acceleration', 'trend_zscore', 'trend_spike', 'trend_strength']
                
                for col in trend_cols_to_merge:
                    if col in trend_df_weekly.columns:
                        # Try different lags to find best alignment
                        best_corr = 0
                        best_lag = 0
                        
                        # Test lags from -4 to +4 weeks
                        for lag in range(-4, 5):
                            temp_df = pd.merge(
                                ticker_datasets[ticker][['returns_20d']],
                                trend_df_weekly[[col]].shift(lag),
                                left_index=True,
                                right_index=True,
                                how='inner'
                            )
                            if len(temp_df) > 20:
                                corr = temp_df['returns_20d'].corr(temp_df[col])
                                if abs(corr) > abs(best_corr):
                                    best_corr = corr
                                    best_lag = lag
                        
                        # Use the best lag
                        ticker_datasets[ticker] = pd.merge(
                            ticker_datasets[ticker],
                            trend_df_weekly[[col]].shift(best_lag).rename(
                                columns={col: f'{col}_{suffix}_lag{best_lag}'}
                            ),
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                        ticker_datasets[ticker][f'{col}_{suffix}_lag{best_lag}'] = \
                            ticker_datasets[ticker][f'{col}_{suffix}_lag{best_lag}'].ffill().fillna(0)
                        merge_count += 1
                        print(f"     ✓ Merged {col}_{suffix} with lag {best_lag} (corr: {best_corr:.3f})")

        print(f"   Total features merged: {merge_count}")

# Train and evaluate models
print("\n🤖 Training models...")
results = {}

for ticker, df in ticker_datasets.items():
    print(f"\n📈 Training model for {ticker}")
    
    # Create target - predict if monthly return is positive
    df['target'] = (df['returns_20d'] > 0).astype(int)
    
    # Check data before dropna
    print(f"   Data shape before final dropna: {df.shape}")
    df = df.dropna()
    print(f"   Data shape after final dropna: {df.shape}")
    
    if len(df) < 100:  # Need at least 100 weeks of data
        print(f"⚠️ Skipping {ticker}: Not enough data ({len(df)} rows).")
        continue
    
        # Select features
    feature_cols = [col for col in df.columns if any(x in col for x in [
        'sma', 'rsi', 'volatility', 'volume_ratio', 'price_to_sma',
        'trend_ma4', 'trend_change_4w', 'trend_momentum', 'trend_acceleration',
        'trend_zscore', 'trend_spike', 'trend_strength'
    ]) and col not in ['returns', 'returns_20d']]  # Exclude target-related columns
    
    print(f"📊 Using {len(feature_cols)} features for {ticker}")
    if len(feature_cols) > 0:
        print(f"   Features: {feature_cols[:5]}...")  # Show first 5 features
    
    if len(feature_cols) == 0:
        print(f"⚠️ Skipping {ticker}: No features available")
        continue
    
    X = df[feature_cols]
    y = df['target']
    
    # Train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Clean infinite values before scaling
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with better parameters for small datasets
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string())
    
    # Identify Google Trends features in top features
    trend_features = feature_importance[feature_importance['feature'].str.contains('trend_')]
    if len(trend_features) > 0:
        print("\n🔍 Google Trends Features Impact:")
        print(trend_features.head().to_string())
    
    # Backtest
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    df['prediction'] = model.predict(scaler.transform(X_clean))
    test_data = df.iloc[split_idx:].copy()
    
    # Calculate strategy returns
    test_data['strategy_returns'] = test_data['prediction'].shift(1) * test_data['returns_20d']
    test_data['cumulative_strategy'] = (1 + test_data['strategy_returns'].fillna(0)).cumprod()
    test_data['cumulative_market'] = (1 + test_data['returns_20d'].fillna(0)).cumprod()
    
    total_return = (test_data['cumulative_strategy'].iloc[-1] - 1) * 100
    market_return = (test_data['cumulative_market'].iloc[-1] - 1) * 100
    
    strategy_mean = test_data['strategy_returns'].mean() * 12  # Annualize
    strategy_std = test_data['strategy_returns'].std() * np.sqrt(12)  # Annualize
    if strategy_std > 0:
        sharpe_ratio = strategy_mean / strategy_std
    else:
        sharpe_ratio = 0
    
    wins = (test_data['strategy_returns'] > 0).sum()
    total_trades = (test_data['prediction'].shift(1) == 1).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    results[ticker] = {
        'AUC': f"{auc_score:.3f}",
        'Strategy Return': f"{total_return:.2f}%",
        'Market Return': f"{market_return:.2f}%", 
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Win Rate': f"{win_rate:.1f}%",
        'Trades': total_trades
    }
    
    print(f"\n📊 Performance Metrics for {ticker}:")
    print(f"Strategy Return: {total_return:.2f}%")
    print(f"Market Return: {market_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Trades: {total_trades}")
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Cumulative returns
    plt.subplot(1, 2, 1)
    plt.plot(test_data.index, test_data['cumulative_market'], label='Buy & Hold', linewidth=2)
    plt.plot(test_data.index, test_data['cumulative_strategy'], label='ML Strategy', linewidth=2)
    plt.title(f"{ticker} Strategy vs Market (Monthly Rebalancing)")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Monthly returns distribution
    plt.subplot(1, 2, 2)
    strategy_returns = test_data['strategy_returns'].dropna()
    market_returns = test_data['returns_20d'].dropna()
    
    plt.hist(market_returns, bins=20, alpha=0.5, label='Market', density=True)
    plt.hist(strategy_returns, bins=20, alpha=0.5, label='Strategy', density=True)
    plt.axvline(market_returns.mean(), color='blue', linestyle='--', label=f'Market Mean: {market_returns.mean():.3f}')
    plt.axvline(strategy_returns.mean(), color='orange', linestyle='--', label=f'Strategy Mean: {strategy_returns.mean():.3f}')
    plt.title('Monthly Returns Distribution')
    plt.xlabel('Monthly Return')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{ticker}_strategy_analysis.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved: {plot_path}")
    plt.close()

# Summary
print("\n" + "="*60)
print("📊 FINAL SUMMARY REPORT")
print("="*60)

if results:
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    
    summary_df.to_csv(os.path.join(plot_dir, 'model_performance_summary.csv'))
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    tickers = list(results.keys())
    
    # Extract numeric values for plotting
    auc_scores = [float(results[t]['AUC']) for t in tickers]
    strategy_returns = [float(results[t]['Strategy Return'].rstrip('%')) for t in tickers]
    market_returns = [float(results[t]['Market Return'].rstrip('%')) for t in tickers]
    sharpe_ratios = [float(results[t]['Sharpe Ratio']) for t in tickers]
    win_rates = [float(results[t]['Win Rate'].rstrip('%')) for t in tickers]
    
    # AUC Scores
    ax = axes[0, 0]
    bars = ax.bar(tickers, auc_scores, color='skyblue')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_ylabel('AUC Score')
    ax.set_title('Model Predictive Power')
    ax.set_ylim(0.4, 0.7)
    for bar, score in zip(bars, auc_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # Returns Comparison
    ax = axes[0, 1]
    x = np.arange(len(tickers))
    width = 0.35
    ax.bar(x - width/2, strategy_returns, width, label='Strategy', color='green', alpha=0.7)
    ax.bar(x + width/2, market_returns, width, label='Market', color='blue', alpha=0.7)
    ax.set_ylabel('Return (%)')
    ax.set_title('Strategy vs Market Returns')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sharpe Ratios
    ax = axes[1, 0]
    bars = ax.bar(tickers, sharpe_ratios, color='orange')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Risk-Adjusted Returns')
    for bar, ratio in zip(bars, sharpe_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{ratio:.2f}', ha='center', va='bottom')
    
    # Win Rates
    ax = axes[1, 1]
    bars = ax.bar(tickers, win_rates, color='purple')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Breakeven (50%)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Strategy Win Rate')
    ax.set_ylim(0, 100)
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    summary_plot_path = os.path.join(plot_dir, 'summary_metrics.png')
    plt.savefig(summary_plot_path)
    print(f"\n✅ Summary visualization saved: {summary_plot_path}")
    plt.close()
    
    print(f"\n✅ All results saved to {plot_dir}/")
else:
    print("❌ No results generated")

print("\n🎉 Analysis complete!")