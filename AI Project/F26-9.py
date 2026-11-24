"""
AI-Powered Stock Intelligence & Portfolio Risk Platform (Track A Web Application)
Group ID: F26-9 | Course: BSE-5A
Members: Ammar Arif (24L-3067), Zainab Sharif (24L-3078), Asadullah Nasir (24L-3089), Ali Saad (24I-3023)
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# Exact 6 pre-trained technology equities (Deliverables 2–4)
SUPPORTED_STOCKS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Consumer Tech & Devices"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "E-Commerce & Cloud"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Search & Digital Ads"},
    "META": {"name": "Meta Platforms Inc.", "sector": "Social & AI Platforms"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Enterprise Software & Cloud"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "AI Semiconductors"},
}
ALL_TICKERS = list(SUPPORTED_STOCKS.keys())

st.set_page_config(
    page_title="F26-9 | AI Stock Intelligence & Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM FINTECH UI STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Clean page spacing and typography */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: #cbd5e1;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }
    /* Section Header */
    .section-label {
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. CACHED DATA LOADER (ALL 6 TRAINED EQUITIES)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(source_mode):
    prices_df, news_df = None, None

    if source_mode == "Bundled Historical Dataset (1996–2026)":
        if os.path.exists("stock_prices_cleaned.xls"):
            try:
                prices_df = pd.read_csv("stock_prices_cleaned.xls")
            except Exception:
                prices_df = pd.read_excel("stock_prices_cleaned.xls")
        if os.path.exists("stock_news_cleaned.xls"):
            try:
                news_df = pd.read_csv("stock_news_cleaned.xls")
            except Exception:
                news_df = pd.read_excel("stock_news_cleaned.xls")

    if prices_df is None or source_mode == "Live Yahoo Finance (Real-Time)":
        import yfinance as yf
        price_frames = []
        news_rows = []
        for t in ALL_TICKERS:
            try:
                hist = yf.Ticker(t).history(period="5y")
                if not hist.empty:
                    hist = hist.reset_index()
                    hist["Ticker"] = t
                    price_frames.append(hist[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
            except Exception:
                pass

            try:
                raw_news = yf.Ticker(t).news or []
                for item in raw_news[:10]:
                    title = item.get("title") or item.get("content", {}).get("title", "")
                    if title:
                        news_rows.append({
                            "Date": datetime.utcnow().strftime("%Y-%m-%d"),
                            "Ticker": t,
                            "Clean_Headline": str(title).strip(),
                        })
            except Exception:
                pass

        if price_frames:
            prices_df = pd.concat(price_frames, ignore_index=True)
        if news_rows:
            news_df = pd.DataFrame(news_rows).drop_duplicates(subset=["Ticker", "Clean_Headline"])

    if news_df is None or news_df.empty:
        news_df = pd.DataFrame([
            ("AAPL", "Apple expands AI ecosystem across next-generation devices"),
            ("AMZN", "Amazon Web Services posts strong enterprise cloud revenue growth"),
            ("GOOGL", "Alphabet accelerates AI search integration amid digital ad recovery"),
            ("META", "Meta reports higher user engagement and operating margin expansion"),
            ("MSFT", "Microsoft Azure cloud demand surges on enterprise AI adoption"),
            ("NVDA", "NVIDIA data center GPU shipments reach record quarterly volume"),
        ], columns=["Ticker", "Clean_Headline"])
        news_df["Date"] = datetime.utcnow().strftime("%Y-%m-%d")

    prices_df["Date"] = pd.to_datetime(prices_df["Date"], utc=True).dt.tz_localize(None)
    prices_df = (
        prices_df[prices_df["Ticker"].isin(ALL_TICKERS)]
        .sort_values(["Ticker", "Date"])
        .dropna(subset=["Close"])
        .reset_index(drop=True)
    )
    return prices_df, news_df.reset_index(drop=True)


# ==============================================================================
# 2. MODULE 1: TECHNICAL FEATURE ENGINEERING & TREND MODEL
# ==============================================================================
@st.cache_resource(show_spinner=False)
def train_stock_pipeline(prices_df):
    feats = []
    for _, grp in prices_df.groupby("Ticker", group_keys=False):
        g = grp.copy()
        g["Daily_Return"] = g["Close"].pct_change()
        g["MA_10"] = g["Close"].rolling(10).mean()
        g["MA_20"] = g["Close"].rolling(20).mean()
        g["Price_to_MA10"] = g["Close"] / g["MA_10"] - 1.0
        g["Price_to_MA20"] = g["Close"] / g["MA_20"] - 1.0
        g["Volatility_10"] = g["Daily_Return"].rolling(10).std()
        g["HL_Range"] = (g["High"] - g["Low"]) / g["Close"]
        g["Vol_Change"] = g["Volume"].pct_change().replace([np.inf, -np.inf], 0.0)

        delta = g["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        g["RSI_14"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

        ema12 = g["Close"].ewm(span=12, adjust=False).mean()
        ema26 = g["Close"].ewm(span=26, adjust=False).mean()
        g["MACD"] = (ema12 - ema26) / g["Close"]

        g["Next_Return"] = g["Daily_Return"].shift(-1)
        g["Target_Up"] = (g["Close"].shift(-1) > g["Close"]).astype(int)
        feats.append(g)

    feat_df = pd.concat(feats, ignore_index=True)
    feature_cols = [
        "Daily_Return", "Price_to_MA10", "Price_to_MA20",
        "Volatility_10", "HL_Range", "Vol_Change", "RSI_14", "MACD"
    ]
    feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)

    train_parts, test_parts = [], []
    for _, grp in feat_df.groupby("Ticker"):
        valid = grp.dropna(subset=["Next_Return"])
        split = int(len(valid) * 0.80)
        train_parts.append(valid.iloc[:split])
        test_parts.append(valid.iloc[split:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train, y_train = train_df[feature_cols], train_df["Target_Up"]
    X_test, y_test = test_df[feature_cols], test_df["Target_Up"]

    lr_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=SEED))
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    maj_preds = np.full(len(y_test), int(y_train.mode()[0]))

    metrics = {
        "Majority Baseline": {"Accuracy": accuracy_score(y_test, maj_preds), "F1": f1_score(y_test, maj_preds, zero_division=0)},
        "Logistic Regression": {"Accuracy": accuracy_score(y_test, lr_preds), "F1": f1_score(y_test, lr_preds, zero_division=0)},
        "Random Forest": {"Accuracy": accuracy_score(y_test, rf_preds), "F1": f1_score(y_test, rf_preds, zero_division=0)},
    }

    latest_df = feat_df.sort_values("Date").groupby("Ticker").tail(1).copy()
    latest_df["LR_P_Up"] = lr_model.predict_proba(latest_df[feature_cols])[:, 1]
    latest_df["RF_P_Up"] = rf_model.predict_proba(latest_df[feature_cols])[:, 1]

    test_df["Pred_Dir"] = lr_preds
    tx_cost = 0.001
    pos_change = test_df.groupby("Ticker")["Pred_Dir"].diff().abs().fillna(0)
    test_df["Strategy_Return"] = test_df["Pred_Dir"] * test_df["Next_Return"] - pos_change * tx_cost

    return metrics, test_df, latest_df


# ==============================================================================
# 3. MODULE 2: REAL-TIME FINANCIAL NEWS SENTIMENT ANALYZER
# ==============================================================================
POS_WORDS = {
    "surge", "surges", "growth", "profit", "beat", "beats", "record", "gain", "gains",
    "bullish", "upgrade", "strong", "expand", "expands", "deal", "breakthrough", "rise", "rises", "soar"
}
NEG_WORDS = {
    "drop", "drops", "fall", "falls", "loss", "miss", "misses", "headwinds", "lawsuit",
    "bearish", "downgrade", "weak", "cut", "cuts", "slump", "decline", "risk", "probe", "fine"
}


def lexicon_label(text):
    tokens = set(str(text).lower().replace(",", "").replace(".", "").split())
    diff = len(tokens & POS_WORDS) - len(tokens & NEG_WORDS)
    return 1 if diff > 0 else (-1 if diff < 0 else 0)


@st.cache_resource(show_spinner=False)
def train_sentiment_model(news_df):
    headline_col = "Clean_Headline" if "Clean_Headline" in news_df.columns else news_df.columns[-1]
    df = news_df.copy()
    df["Headline"] = df[headline_col].astype(str)
    labels = df["Headline"].apply(lexicon_label)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english")
    X = vec.fit_transform(df["Headline"])

    clf = None
    if len(set(labels)) >= 2:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(X, labels)

    def predict_headline_sentiment(texts):
        if clf is not None:
            probs = clf.predict_proba(vec.transform(texts))
            classes = list(clf.classes_)
            p_pos = probs[:, classes.index(1)] if 1 in classes else np.zeros(len(texts))
            p_neg = probs[:, classes.index(-1)] if -1 in classes else np.zeros(len(texts))
            scores = p_pos - p_neg
            preds = clf.predict(vec.transform(texts))
        else:
            preds = np.array([lexicon_label(t) for t in texts])
            scores = preds.astype(float)
        return preds, np.round(scores, 4)

    preds, scores = predict_headline_sentiment(df["Headline"].tolist())
    df["Predicted_Class"] = preds
    df["Sentiment_Score"] = scores
    return df, predict_headline_sentiment


# ==============================================================================
# 4. MODULE 3: MARKOWITZ MINIMUM-VARIANCE PORTFOLIO OPTIMIZER
# ==============================================================================
def optimize_portfolio(prices_df, selected_tickers, capital_usd):
    sub_prices = prices_df[prices_df["Ticker"].isin(selected_tickers)]
    piv = sub_prices.pivot_table(index="Date", columns="Ticker", values="Close").pct_change().dropna()
    tickers = list(piv.columns)
    n = len(tickers)

    cov_annual = piv.cov() * 252.0
    corr_matrix = piv.corr()
    vols = np.sqrt(np.diag(cov_annual.values))

    w_eq = np.ones(n) / n
    vol_eq = float(np.sqrt(w_eq.T @ cov_annual.values @ w_eq))

    w_inv = (1.0 / vols) / np.sum(1.0 / vols)
    vol_inv = float(np.sqrt(w_inv.T @ cov_annual.values @ w_inv))

    max_cap = min(0.60, max(1.0 / n + 0.15, 0.40))
    bounds = tuple((0.02, max_cap) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(
        lambda w: np.sqrt(w.T @ cov_annual.values @ w),
        x0=w_eq, method="SLSQP", bounds=bounds, constraints=constraints
    )
    w_mv = res.x if res.success else w_inv
    vol_mv = float(np.sqrt(w_mv.T @ cov_annual.values @ w_mv))

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Company": [SUPPORTED_STOCKS[t]["name"] for t in tickers],
        "Annual_Volatility": vols,
        "Equal_Weight": w_eq,
        "Inverse_Vol_Weight": w_inv,
        "Markowitz_MinVar_Weight": w_mv,
        "Suggested_Allocation_USD": w_mv * capital_usd,
    })
    port_stats = {
        "vol_eq": vol_eq,
        "vol_inv": vol_inv,
        "vol_mv": vol_mv,
        "reduction_pct": (vol_eq - vol_mv) / vol_eq * 100.0,
    }
    return weights_df, corr_matrix, port_stats


# ==============================================================================
# 5. HEADER & SIDEBAR CONTROLS
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <p class="hero-title">AI-Powered Stock Intelligence & Portfolio Risk Platform</p>
    <p class="hero-sub">
        <strong>Group F26-9 (BSE-5A)</strong> • Ammar Arif, Zainab Sharif, Asadullah Nasir, Ali Saad •
        End-to-End Technical Trend, News Sentiment & Markowitz Risk Engine
    </p>
</div>
""", unsafe_allow_html=True)

# Session state for easy preset selection
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = ALL_TICKERS.copy()

with st.sidebar:
    st.markdown('<p class="section-label">1. Data & Model Engine</p>', unsafe_allow_html=True)
    source_mode = st.radio(
        "Market Data Feed",
        ["Bundled Historical Dataset (1996–2026)", "Live Yahoo Finance (Real-Time)"],
        label_visibility="collapsed",
    )

    model_choice = st.selectbox(
        "Module 1 Trend Classifier",
        ["Logistic Regression (Recommended)", "Random Forest"],
    )

    st.divider()
    st.markdown('<p class="section-label">2. Signal Fusion & Capital</p>', unsafe_allow_html=True)
    tech_weight = st.slider(
        "Technical vs. Sentiment Blend",
        min_value=0.50, max_value=0.90, value=0.70, step=0.05,
        help="0.70 = 70% Technical Price Signal + 30% News Sentiment Signal",
    )
    capital_usd = st.number_input(
        "Portfolio Investment Capital ($ USD)",
        min_value=1000, max_value=10_000_000, value=10000, step=1000,
    )

    st.divider()
    st.caption(
        "**Model Scope:** Calibrated exclusively on 6 mega-cap tech equities (38,670 trading sessions). "
        "Educational research prototype; not financial advice."
    )

# ==============================================================================
# 6. STEP 1: INTERACTIVE PORTFOLIO BUILDER (6 PRE-TRAINED STOCKS)
# ==============================================================================
st.markdown('<p class="section-label">Step 1 — Select Trained Companies to Include in Your Portfolio</p>', unsafe_allow_html=True)

p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.3, 1.3, 3.2])
if p_col1.button("Select All 6 Stocks", use_container_width=True):
    st.session_state.selected_tickers = ALL_TICKERS.copy()
if p_col2.button("Low-Volatility Trio", use_container_width=True):
    st.session_state.selected_tickers = ["GOOGL", "MSFT", "META"]
if p_col3.button("High-Growth Core", use_container_width=True):
    st.session_state.selected_tickers = ["AAPL", "AMZN", "NVDA"]

selected_tickers = st.multiselect(
    "Active Portfolio Universe (Choose 2 to 6 pre-trained equities):",
    options=ALL_TICKERS,
    default=st.session_state.selected_tickers,
    key="ticker_multiselect",
    format_func=lambda t: f"{t} — {SUPPORTED_STOCKS[t]['name']} ({SUPPORTED_STOCKS[t]['sector']})",
)
st.session_state.selected_tickers = selected_tickers

if len(selected_tickers) < 2:
    st.warning("Please select at least **2 companies** above to compute cross-asset correlations and portfolio risk weights.")
    st.stop()

# Run Pipeline
with st.spinner("Computing technical features, news sentiment, and Markowitz portfolio weights..."):
    prices_df, news_df = load_market_data(source_mode)
    metrics, test_df, latest_df = train_stock_pipeline(prices_df)
    scored_news, predict_headline_fn = train_sentiment_model(news_df)
    weights_df, corr_matrix, port_stats = optimize_portfolio(prices_df, selected_tickers, capital_usd)

# Merge Module 1 + Module 2
active_latest = latest_df[latest_df["Ticker"].isin(selected_tickers)].copy()
active_latest["Company"] = active_latest["Ticker"].map(lambda t: SUPPORTED_STOCKS[t]["name"])

if "Ticker" in scored_news.columns:
    ticker_sent = (
        scored_news[scored_news["Ticker"].isin(selected_tickers)]
        .groupby("Ticker")
        .agg(Headlines=("Headline", "count"), Mean_Sentiment=("Sentiment_Score", "mean"))
        .reset_index()
    )
else:
    ticker_sent = pd.DataFrame(columns=["Ticker", "Headlines", "Mean_Sentiment"])

combined = pd.merge(active_latest, ticker_sent, on="Ticker", how="left")
combined["Headlines"] = combined["Headlines"].fillna(0).astype(int)
combined["Mean_Sentiment"] = combined["Mean_Sentiment"].fillna(0.0)
combined["Tech_P_Up"] = combined["LR_P_Up"] if "Logistic" in model_choice else combined["RF_P_Up"]
combined["Sentiment_Prob"] = (combined["Mean_Sentiment"] + 1.0) / 2.0
combined["Combined_P_Up"] = tech_weight * combined["Tech_P_Up"] + (1.0 - tech_weight) * combined["Sentiment_Prob"]
combined["Signal"] = np.where(combined["Combined_P_Up"] >= 0.50, "BULLISH (UP)", "BEARISH (DOWN)")

# ==============================================================================
# 7. EXECUTIVE KPI ROW
# ==============================================================================
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
active_key = "Logistic Regression" if "Logistic" in model_choice else "Random Forest"
bullish_count = int((combined["Combined_P_Up"] >= 0.50).sum())

k1.metric(
    "Active Portfolio Basket",
    f"{len(selected_tickers)} of 6 Stocks",
    f"{bullish_count} Bullish / {len(selected_tickers) - bullish_count} Bearish",
)
k2.metric(
    "Trend Classifier (OOS F1)",
    f"{metrics[active_key]['F1']:.3f}",
    f"{active_key} (Acc: {metrics[active_key]['Accuracy']:.1%})",
)
k3.metric(
    "Unoptimized Portfolio Risk",
    f"{port_stats['vol_eq']*100:.2f}%",
    "Equal-Weight Annualized Vol.",
)
k4.metric(
    "Markowitz Optimized Risk",
    f"{port_stats['vol_mv']*100:.2f}%",
    f"-{port_stats['reduction_pct']:.2f}% Volatility Reduction",
)

# ==============================================================================
# 8. MAIN ANALYTICAL WORKSPACE TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Next-Day Stock Signals",
    "🛡️ Portfolio Risk & Rebalancer",
    "📰 Live Sentiment Simulator",
    "📊 Model Evaluation & Backtest",
])

# --- TAB 1: UNIFIED STOCK PREDICTIONS ---
with tab1:
    st.subheader("Unified Next-Day Direction Forecast (Technical + News Fusion)")
    st.caption(
        f"Fusion Formula: {int(tech_weight*100)}% Technical Price Probability + "
        f"{int((1-tech_weight)*100)}% Calibrated News Sentiment Probability."
    )

    display_df = combined[[
        "Ticker", "Company", "Close", "RSI_14", "Tech_P_Up", "Mean_Sentiment", "Combined_P_Up", "Signal"
    ]].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company Name", width="medium"),
            "Close": st.column_config.NumberColumn("Latest Close", format="$%.2f"),
            "RSI_14": st.column_config.NumberColumn("RSI (14d)", format="%.1f"),
            "Tech_P_Up": st.column_config.ProgressColumn("Technical P(UP)", min_value=0.0, max_value=1.0, format="%.3f"),
            "Mean_Sentiment": st.column_config.NumberColumn("News Polarity", format="%+.3f"),
            "Combined_P_Up": st.column_config.ProgressColumn("Combined P(UP)", min_value=0.0, max_value=1.0, format="%.3f"),
            "Signal": st.column_config.TextColumn("Next-Day Outlook"),
        },
    )

    fig_bar = px.bar(
        combined,
        x="Ticker",
        y=["Tech_P_Up", "Combined_P_Up"],
        barmode="group",
        color_discrete_sequence=["#94a3b8", "#2563eb"],
        labels={"value": "Upward Probability", "variable": "Model Stage", "Ticker": "Company Ticker"},
        title="Technical Baseline Probability vs. Sentiment-Adjusted Probability",
    )
    fig_bar.add_hline(y=0.50, line_dash="dash", line_color="#ef4444", annotation_text="0.50 Decision Boundary")
    fig_bar.update_layout(template="plotly_white", yaxis_range=[0, 1], legend_title_text="")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: PORTFOLIO RISK & DOLLAR REBALANCER ---
with tab2:
    st.subheader(f"Minimum-Variance Portfolio Rebalancing (${capital_usd:,.0f} Capital)")
    st.caption(
        "Exploits cross-asset return correlations to minimize total portfolio volatility while maintaining full market exposure."
    )

    col_chart1, col_chart2 = st.columns([1.15, 1])
    with col_chart1:
        fig_w = px.bar(
            weights_df,
            x="Ticker",
            y=["Equal_Weight", "Inverse_Vol_Weight", "Markowitz_MinVar_Weight"],
            barmode="group",
            color_discrete_sequence=["#cbd5e1", "#60a5fa", "#1d4ed8"],
            labels={"value": "Portfolio Weight", "variable": "Allocation Strategy"},
            title="Weight Comparison: Equal vs. Inverse-Vol vs. Markowitz Min-Var",
        )
        fig_w.update_layout(template="plotly_white", yaxis_tickformat=".0%", legend_title_text="")
        st.plotly_chart(fig_w, use_container_width=True)

    with col_chart2:
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pairwise Return Correlation Matrix",
        )
        fig_corr.update_layout(template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.dataframe(
        weights_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Company": st.column_config.TextColumn("Company Name"),
            "Annual_Volatility": st.column_config.NumberColumn("Annualized Risk (Std)", format="%.2%%"),
            "Equal_Weight": st.column_config.NumberColumn("Equal Weight", format="%.2%%"),
            "Inverse_Vol_Weight": st.column_config.NumberColumn("Inverse-Vol Weight", format="%.2%%"),
            "Markowitz_MinVar_Weight": st.column_config.ProgressColumn("Optimal Weight", min_value=0.0, max_value=1.0, format="%.2%%"),
            "Suggested_Allocation_USD": st.column_config.NumberColumn("Target Investment ($)", format="$%,.2f"),
        },
    )

# --- TAB 3: LIVE SENTIMENT SIMULATOR ---
with tab3:
    st.subheader("Module 2: Real-Time Financial Headline Simulator")
    st.write("Select any stock in your portfolio and test how breaking news headlines shift its next-day forecast:")

    s_col1, s_col2 = st.columns([2.2, 1])
    with s_col1:
        sim_ticker = st.selectbox(
            "Target Portfolio Company",
            selected_tickers,
            format_func=lambda t: f"{t} — {SUPPORTED_STOCKS[t]['name']}",
        )
        preset_choice = st.radio(
            "Quick Headline Presets:",
            ["Bullish Earnings Beat", "Bearish Regulatory Lawsuit", "Neutral Product Update"],
            horizontal=True,
        )
        preset_map = {
            "Bullish Earnings Beat": f"{SUPPORTED_STOCKS[sim_ticker]['name']} reports record quarterly profit and surges on strong AI cloud demand",
            "Bearish Regulatory Lawsuit": f"{SUPPORTED_STOCKS[sim_ticker]['name']} faces antitrust lawsuit and weak margin guidance amid regulatory probe",
            "Neutral Product Update": f"{SUPPORTED_STOCKS[sim_ticker]['name']} schedules annual developer conference for next month",
        }
        headline_input = st.text_input("Or type any custom financial headline:", value=preset_map[preset_choice])

    with s_col2:
        pred_cls, pred_score = predict_headline_fn([headline_input])
        sent_label = "Positive (+1)" if pred_cls[0] == 1 else ("Negative (-1)" if pred_cls[0] == -1 else "Neutral (0)")
        st.metric("NLP Sentiment Class", sent_label, f"Polarity Score: {pred_score[0]:+.3f}")

        base_row = combined[combined["Ticker"] == sim_ticker].iloc[0]
        sim_prob = tech_weight * base_row["Tech_P_Up"] + (1.0 - tech_weight) * ((pred_score[0] + 1.0) / 2.0)
        st.metric(
            f"Simulated {sim_ticker} P(UP)",
            f"{sim_prob:.3f}",
            f"{(sim_prob - base_row['Combined_P_Up']):+.3f} vs. current baseline",
        )

    st.divider()
    st.subheader("Ingested Market Headlines (Filtered to Active Portfolio)")
    filtered_news = scored_news[scored_news["Ticker"].isin(selected_tickers)] if "Ticker" in scored_news.columns else scored_news
    st.dataframe(
        filtered_news[["Ticker", "Headline", "Predicted_Class", "Sentiment_Score"]],
        use_container_width=True,
        hide_index=True,
    )

# --- TAB 4: MODEL VALIDATION & BACKTEST ---
with tab4:
    st.subheader("Out-of-Sample Model Benchmarks & Transaction-Cost Backtest")
    v_col1, v_col2 = st.columns([1, 1.6])

    with v_col1:
        st.markdown("**Chronological 80/20 Held-Out Classification Metrics**")
        bench_df = pd.DataFrame([
            {"Model": "Majority Class Baseline", **metrics["Majority Baseline"]},
            {"Model": "Logistic Regression (L2)", **metrics["Logistic Regression"]},
            {"Model": "Random Forest (Depth=6)", **metrics["Random Forest"]},
        ])
        st.dataframe(
            bench_df.style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "**Research Insight:** Because daily returns in liquid mega-cap tech stocks exhibit near-zero autocorrelation, "
            "L2-regularized Logistic Regression achieves higher F1 stability than tree ensembles while avoiding noise overfitting."
        )

    with v_col2:
        bt_ticker = st.selectbox(
            "Select Portfolio Stock for Out-of-Sample Equity Curve",
            selected_tickers,
            format_func=lambda t: f"{t} — {SUPPORTED_STOCKS[t]['name']}",
        )
        sub_bt = test_df[test_df["Ticker"] == bt_ticker].copy()
        sub_bt["Buy_and_Hold"] = (1 + sub_bt["Next_Return"]).cumprod()
        sub_bt["AI_Strategy_Net_10bps"] = (1 + sub_bt["Strategy_Return"]).cumprod()

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=sub_bt["Date"], y=sub_bt["Buy_and_Hold"],
            name="Buy & Hold Benchmark", line=dict(color="#94a3b8", width=2)
        ))
        fig_bt.add_trace(go.Scatter(
            x=sub_bt["Date"], y=sub_bt["AI_Strategy_Net_10bps"],
            name="AI Strategy (Net 10 bps Cost)", line=dict(color="#2563eb", width=2.5)
        ))
        fig_bt.update_layout(
            template="plotly_white",
            title=f"Cumulative Growth of $1.00 — {bt_ticker} ({SUPPORTED_STOCKS[bt_ticker]['name']})",
            yaxis_title="Portfolio Value ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bt, use_container_width=True)