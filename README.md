#  AI-Powered Stock Intelligence & Portfolio Risk Platform

##  Overview

Traditional retail investment tools often isolate individual technical indicators from breaking news sentiment or evaluate single-stock predictions without accounting for portfolio-level systemic risk. 

This project addresses that gap by building an end-to-end, three-stage connected pipeline deployed as an interactive **Streamlit** web application:
1. **Module 1 — Technical Trend Predictor:** Engineers leakage-free technical indicators (`Daily_Return`, `MA_10`, `MA_20`, `RSI_14`, `MACD`, `Volatility_10`, `HL_Range`) to estimate next-day upward price probability $P_{\text{tech}}(\text{UP})$.
2. **Module 2 — Financial News Sentiment Engine:** Applies TF-IDF unigram/bigram vectorization with class-balanced Logistic Regression to score recent company headlines into continuous polarity $s \in [-1, 1]$, normalized to probability space via $P_{\text{sent}} = (s + 1)/2$.
3. **Signal Fusion & Module 3 — Markowitz Minimum-Variance Optimizer:** Fuses technical and sentiment probabilities ($P_{\text{combined}} = \alpha P_{\text{tech}} + (1 - \alpha) P_{\text{sent}}$) and solves constrained quadratic minimum-variance optimization ($\min_w \sqrt{w^T \Sigma w}$) across the user's selected stock basket to lower overall portfolio volatility.

---

##  Supported Stock Universe

Our models are calibrated specifically on six mega-cap technology equities spanning **38,670 historical trading sessions (1996–2026)**:

| Ticker | Company Name | Sector |
| :--- | :--- | :--- |
| **AAPL** | Apple Inc. | Consumer Tech & Devices |
| **AMZN** | Amazon.com Inc. | E-Commerce & Cloud |
| **GOOGL** | Alphabet Inc. | Search & Digital Ads |
| **META** | Meta Platforms Inc. | Social & AI Platforms |
| **MSFT** | Microsoft Corp. | Enterprise Software & Cloud |
| **NVDA** | NVIDIA Corp. | AI Semiconductors |

---

##  Web Application Features

* **Interactive Portfolio Builder:** Select any subset of 2 to 6 pre-trained companies (or use one-click presets) and enter custom investment capital (USD) to compute exact dollar allocations.
* **Dual Data Modes:** Switch seamlessly between the **Bundled Historical Dataset (1996–2026)** for 100% reproducible results and **Live Yahoo Finance (Real-Time)** for up-to-date market inference.
* **Tab 1 — Next-Day Stock Signals:** Displays latest closing price, 14-day RSI, standalone technical probability, news sentiment polarity, fused probability, and directional outlook (`BULLISH (UP)` vs. `BEARISH (DOWN)`).
* **Tab 2 — Portfolio Risk & Rebalancer:** Visualizes the cross-asset return correlation matrix and compares **Equal-Weight**, **Inverse-Volatility**, and **Markowitz Minimum-Variance** allocations alongside expected annualized risk reduction.
* **Tab 3 — Live Sentiment Simulator:** Test Module 2 in real time by entering custom financial headlines (or selecting presets) to see how breaking news shifts next-day probabilities.
* **Tab 4 — Model Evaluation & Backtest:** Compares **Logistic Regression (L2)** and **Random Forest** against a **Majority Class Baseline** on a chronological 80/20 split, including a transaction-cost-adjusted (10 bps) cumulative return curve vs. Buy-and-Hold.

---

##  Repository Structure

```text
AI-Powered-Stock-Intelligence-Risk-Platform/
├── F26-9.py                     # Main Streamlit web application & 3-module ML pipeline
├── requirements.txt             # Python dependencies
├── stock_prices_cleaned.xls     # Preprocessed historical OHLCV dataset (38,670 records)
├── stock_news_cleaned.xls       # Sanitized financial news headlines dataset
└── README.md                    # Project documentation
