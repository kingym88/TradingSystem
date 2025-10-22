# TradingSystem: Two-Stage Enhanced ML AI Trading Platform

An advanced multi-stage AI trading platform for stock selection, portfolio management, and risk analytics in US equity markets. This system leverages live market data, robust technical analysis, Bayesian ML methods, and scalable portfolio tracking.

## 🚀 System Highlights

- **Two-Stage Stock Selection:**  
  - **Stage 1:** Rapidly filters a universe of 4000+ US-listed stocks using market cap, price, liquidity, and basic screening to select top candidates for detailed analysis.
  - **Stage 2:** Executes deep technical and momentum analysis (RSI, MACD, ADX, SMA, Bollinger Bands, volatility, sentiment, etc.), then ranks and scores stocks for actionable recommendations.

- **ML-Powered Recommendation Engine:**  
  - Uses Random Forest, XGBoost, and Bayesian learning from trade outcomes.
  - Confidence intervals and win rates are adapted with Bayesian updating from your historical trades.
  - Recommendations are split into: portfolio actions for stocks you own (buy/sell/hold) and new high-confidence opportunities.

- **Portfolio Management & Analytics:**  
  - Tracks cash, invested capital, current market value, live prices, returns, P&L, and risk.
  - Implements advanced Kelly criterion-based position sizing, stop-loss, and take-profit management.
  - Database layer (SQLAlchemy) supports trade logs, portfolio snapshots, AI recommendations, and historical analysis.

- **Trade Logger with Bulk Import:**  
  - Supports interactive trade entry, CSV bulk import, quick manual pasting, and multi-trade batch logging.
  - Built-in trade validation, error reporting, export to CSV, and template generation.

- **Integrated API Framework:**  
  - Yahoo Finance for live prices and historical data (yfinance)
  - News, sentiment, and alternative data (VADER, Transformers, AlphaVantage, NewsAPI, etc., optional)
  - Perplexity API integration for AI-powered strategic suggestions (set API key in `.env`)

- **Advanced Reporting & Analysis:**  
  - Performance analyzer for returns, trade metrics, portfolio changes, and summary statistics.
  - Deep-dive weekend analyzer for portfolio review, ML insights, strategic recommendations, and market condition analysis.

## 🛠️ Folder Structure

- `database.py` – SQLAlchemy data models, trade/portfolio/logging persistence
- `two_stage_trading_system.py` – Main runner and CLI for two-stage analysis, execution, and simulation
- `two_stage_data_manager.py` – Data ingestion, Stage 1 & 2 selection, technical and sentiment calculation
- `two_stage_ml_engine.py` – ML model training, Bayesian logic, final recommendation generation
- `portfolio_manager.py` – Position tracking, live price update, Kelly sizing, and trade validation
- `trade_logger.py` – Interactive and bulk trade logger, import/export, error handler
- `perplexity_client.py` – AI API wrapper for strategy recommendations and market research
- `performance_analyzer.py` – Analysis and summary reporting for returns/trades/portfolio
- `weekend_analyzer.py` – Automated weekend deep analysis utility
- `two_stage_config.py` – Central config: API keys, risk limits, ML parameters, screening
- `enhanced_requirements.txt` – Full requirements for dependencies (see below)
- `.env.template`, `.gitattributes`, setup scripts – Environment and platform setup files

## ⚙️ Setup

1. **Clone Repository**
   
2. **Install Dependencies**
- pip install -r enhanced_requirements.txt
   
3. **Environment Configuration**  
- Copy `.env.template` to `.env`
- Add your API keys (Perplexity, Yahoo, AlphaVantage, etc.)

4. **Run Main System**
- python two_stage_trading_system.py

## 🏄 Workflow

- Run daily two-stage analysis for recommendations.
- Log trades in real-time, interactively or in bulk.
- Monitor portfolio and performance analytics.
- Use the weekend analyzer for strategic insights.
- Export/import trades as CSV and review trade summaries.

## 🎯 Key Algorithms

- **Two-Stage Filtering:** Smart, parallel batch screening and deep technical scoring.
- **ML Confidence:** Blend of statistical/bayesian learning, ML calibration (scikit-learn/XGBoost), and user feedback.
- **Position Sizing:** Kelly criterion, risk limits, trade-level stop-loss/take-profit, cash tracking.
- **Sentiment Analysis:** Optional—news and NLP-driven sentiment scores on candidates.
- **Database Persistence:** Detailed logs for all trading activity, recommendations, and portfolio changes.

## 📊 Reporting Features

- Trade summary (recent, comprehensive, by symbol)
- Portfolio and position metrics (cash, value, returns, positions)
- Performance analytics (return, drawdown, trade success metrics)
- Automated export and template creation

## 💡 Unique Aspects

- Robust technical indicators and volatility/momentum analytics
- Dynamic ML learning from trade history
- Smart trade simulation and validation
- Scalable to thousands of stocks with parallel processing
- Modular design — components extensible for research, live trading, or purely analytical

## ⚠️ Disclaimer

- **No live trading:** Analysis and recommendations only; execution is simulated or manual.
- **Educational Purpose:** For learning and research; responsibility for trading decisions rests with user.
- **API keys required for full feature set.**

## 📦 Requirements

See `enhanced_requirements.txt` for full list of required Python packages.  
Most common ML, data-science, and fin-API libraries included (pandas, numpy, sklearn, xgboost, yfinance, SQLAlchemy, pydantic, transformers, matplotlib, loguru, etc.).

## 📝 Author

Built and maintained by [kingym88](https://github.com/kingym88).  
Inspired by [ChatGPT-Micro-Cap-Experiment](https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment).


