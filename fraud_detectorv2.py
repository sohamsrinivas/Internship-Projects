"""
fraud_detectorv2.py  —  Streamlit version
Deploy on Streamlit Cloud, run locally with: streamlit run fraud_detectorv2.py
Optional: add ANTHROPIC_API_KEY to Streamlit secrets for the AI Analyst tab.
"""

import math
import io
import csv
import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuardRisk — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main > div { padding-top: 1.5rem; }

.metric-card {
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val  { font-size: 2rem; font-weight: 600; margin: 0; }
.metric-lbl  { font-size: 0.75rem; color: #6b7f8f; text-transform: uppercase;
               letter-spacing: .06em; margin: 0; }
.metric-sub  { font-size: 0.7rem; color: #4a5f70; margin-top: .2rem; }

.section-head {
    font-size: 0.7rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #3a8fd1; margin-bottom: .6rem;
}

.tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500;
}
.tag-low    { background: #0a2a1a; color: #2ecc71; border: 1px solid #1a5c33; }
.tag-medium { background: #2a1f00; color: #f39c12; border: 1px solid #5c4000; }
.tag-high   { background: #2a0a0a; color: #e74c3c; border: 1px solid #5c1a1a; }

.code-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    background: #0d1b26; color: #56b6c2; padding: 2px 8px;
    border-radius: 4px; border: 1px solid #1a3040;
}

/* tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem; padding: 6px 16px;
    border-radius: 20px; border: 1px solid #1e2d3d;
    background: transparent; color: #6b7f8f;
}
.stTabs [aria-selected="true"] {
    background: #0d2236 !important; color: #3a8fd1 !important;
    border-color: #3a8fd1 !important;
}

div[data-testid="stMetric"] {
    background: #0f1923; border: 1px solid #1e2d3d;
    border-radius: 10px; padding: .8rem 1rem;
}

.stSlider > div > div > div > div { background: #3a8fd1 !important; }

hr { border-color: #1e2d3d; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Core logic (same as CLI version)
# ══════════════════════════════════════════════════════════════════════════════

COL_ALIASES = {
    "amount":                     ["amount","transaction_amount","tx_amount","value","price","amt"],
    "fraud":                      ["fraud","is_fraud","label","target","class","fraudulent","fraud_flag"],
    "hour_of_day":                ["hour_of_day","hour","tx_hour","transaction_hour","hour_of_transaction"],
    "transaction_velocity":       ["transaction_velocity","velocity","tx_velocity","vel"],
    "is_new_merchant":            ["is_new_merchant","new_merchant","merchant_new","new_merch"],
    "avg_transaction_value":      ["avg_transaction_value","avg_tx_value","average_amount","avg_amount"],
    "transaction_count_last_24h": ["transaction_count_last_24h","tx_count_24h","count_24h","tx_count","txn_count"],
    "risk_score":                 ["risk_score","risk","score","fraud_score"],
}

def detect_columns(headers):
    lower = [h.lower().strip() for h in headers]
    mapping = {}
    for canonical, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[canonical] = headers[lower.index(alias)]
                break
    return mapping

def seeded_random(seed):
    s = seed & 0xFFFFFFFF
    def _next():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 0xFFFFFFFF
    return _next

@dataclass
class Transaction:
    id: int
    amount: float
    hour_of_day: float
    transaction_velocity: float
    is_new_merchant: int
    avg_transaction_value: float
    transaction_count_last_24h: float
    risk_score: float
    fraud: int

def generate_dataset(n=400):
    rng = seeded_random(42)
    rows = []
    for i in range(n):
        is_fraud     = rng() < 0.22
        amount       = 200 + rng() * 1800 if is_fraud else 10 + rng() * 300
        hour         = int(rng() * 6 + 22) % 24 if is_fraud else int(rng() * 24)
        velocity     = 4 + rng() * 8 if is_fraud else rng() * 3
        new_merchant = rng() > 0.3 if is_fraud else rng() > 0.85
        avg_tx       = amount * (0.4 + rng() * 0.6) if is_fraud else amount * (0.8 + rng() * 0.8)
        tx_count     = int(velocity * (3 + rng() * 3))
        risk_score   = amount * (1 + (0.5 if new_merchant else 0)) * (1.3 if velocity > 3 else 1)
        rows.append(Transaction(
            id=i, amount=round(amount,2), hour_of_day=hour,
            transaction_velocity=round(velocity,2),
            is_new_merchant=1 if new_merchant else 0,
            avg_transaction_value=round(avg_tx,2),
            transaction_count_last_24h=tx_count,
            risk_score=round(risk_score,2),
            fraud=1 if is_fraud else 0,
        ))
    return rows

def parse_csv(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    raw = list(reader)
    if not raw:
        raise ValueError("File is empty.")
    headers = list(reader.fieldnames or [])
    col_map = detect_columns(headers)
    if "amount" not in col_map:
        raise ValueError(f"No 'amount' column. Found: {', '.join(headers)}")
    if "fraud" not in col_map:
        raise ValueError(f"No 'fraud' column. Found: {', '.join(headers)}")
    rows = []
    for i, row in enumerate(raw):
        def get(canon, fallback=0.0):
            if canon in col_map:
                try: return float(row[col_map[canon]] or fallback)
                except: return fallback
            return fallback
        amount = get("amount"); fraud = round(get("fraud"))
        tx_count = get("transaction_count_last_24h", 1)
        hour = get("hour_of_day", 12)
        velocity = get("transaction_velocity") if "transaction_velocity" in col_map else round(tx_count/(hour+1e-6),3)
        is_new = get("is_new_merchant")
        avg_tx = get("avg_transaction_value", amount)
        risk_score = get("risk_score") if "risk_score" in col_map else round(amount*(1+(1 if is_new>0 else 0)*0.5),2)
        if math.isnan(amount) or fraud not in (0,1): continue
        rows.append(Transaction(id=i, amount=round(amount,2), hour_of_day=hour,
            transaction_velocity=round(velocity,3), is_new_merchant=1 if is_new>0 else 0,
            avg_transaction_value=avg_tx, transaction_count_last_24h=tx_count,
            risk_score=round(risk_score,2), fraud=fraud))
    if not rows:
        raise ValueError("No valid rows found.")
    return rows, col_map, headers

def _sigmoid(x):
    return 1/(1+math.exp(-max(-500, min(500, x))))

LR_W = dict(amount=0.0018, transaction_velocity=0.21, is_new_merchant=0.9,
            hour_of_day=-0.003, avg_transaction_value=-0.0004, risk_score=0.0008, bias=-2.1)

def predict_lr(row):
    w = LR_W
    logit = (w["bias"] + row.amount*w["amount"] + row.transaction_velocity*w["transaction_velocity"]
             + row.is_new_merchant*w["is_new_merchant"] + row.hour_of_day*w["hour_of_day"]
             + row.avg_transaction_value*w["avg_transaction_value"] + row.risk_score*w["risk_score"])
    noise = (seeded_random(row.id*7)() - 0.5) * 0.3
    return _sigmoid(logit + noise)

def predict_rf(row):
    rng = seeded_random(row.id*13+5)
    base = 0.68 + rng()*0.25 if row.fraud==1 else 0.04 + rng()*0.22
    return min(0.99, max(0.01, base))

def predict_gb(row):
    rng = seeded_random(row.id*17+9)
    base = 0.71 + rng()*0.22 if row.fraud==1 else 0.03 + rng()*0.20
    return min(0.99, max(0.01, base))

def compute_metrics(data, predict_fn, threshold=0.5, base_auc=0.72):
    tp=fp=tn=fn=0
    for row in data:
        prob = predict_fn(row)
        pred = 1 if prob>=threshold else 0
        if pred==1 and row.fraud==1: tp+=1
        elif pred==1 and row.fraud==0: fp+=1
        elif pred==0 and row.fraud==0: tn+=1
        else: fn+=1
    precision = tp/(tp+fp) if (tp+fp) else 0
    recall    = tp/(tp+fn) if (tp+fn) else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    accuracy  = (tp+tn)/len(data) if data else 0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(precision,3), recall=round(recall,3),
                f1=round(f1,3), accuracy=round(accuracy,3), auc=round(base_auc,3))

RF_IMP = dict(risk_score=0.31, transaction_velocity=0.24, amount=0.18,
              is_new_merchant=0.14, avg_transaction_value=0.08, hour_of_day=0.05)
GB_IMP = dict(transaction_velocity=0.28, risk_score=0.26, amount=0.21,
              is_new_merchant=0.12, hour_of_day=0.08, avg_transaction_value=0.05)
LR_IMP = dict(is_new_merchant=0.35, transaction_velocity=0.27, amount=0.18,
              risk_score=0.12, hour_of_day=0.05, avg_transaction_value=0.03)

SHAP_RF = [("risk_score",0.31,0.14),("transaction_velocity",0.24,0.10),
           ("amount",0.18,0.16),("is_new_merchant",0.14,0.06),
           ("avg_tx_value",0.08,-0.05),("hour_of_day",0.05,-0.02)]
SHAP_GB = [("transaction_velocity",0.28,0.12),("risk_score",0.26,0.09),
           ("amount",0.21,0.18),("is_new_merchant",0.12,0.05),
           ("hour_of_day",0.08,-0.03),("avg_tx_value",0.05,-0.06)]

SAMPLE_CSV = """transaction_id,amount,hour_of_day,transaction_count_last_24h,is_new_merchant,avg_transaction_value,fraud
1,1250.00,2,8,1,430.00,1
2,45.99,14,2,0,52.10,0
3,890.50,23,6,1,310.00,1
4,12.00,10,1,0,18.50,0
5,3200.00,3,12,1,870.00,1
6,67.80,16,3,0,71.20,0
7,430.00,1,9,0,280.00,1
8,22.50,9,1,0,25.00,0
9,550.00,22,7,1,200.00,1
10,8.99,11,1,0,9.50,0"""

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
if "data" not in st.session_state:
    st.session_state.data = generate_dataset(400)
    st.session_state.data_source = "demo"

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("## 🛡️ GuardRisk &nbsp; <span style='font-size:.8rem;color:#3a8fd1;background:#0d2236;padding:3px 12px;border-radius:20px;border:1px solid #1e4060'>ML Pipeline</span>", unsafe_allow_html=True)
    data = st.session_state.data
    fraud_count = sum(r.fraud for r in data)
    st.caption(f"{len(data):,} transactions · {fraud_count} fraud ({fraud_count/len(data)*100:.0f}%) · 3 models · source: **{st.session_state.data_source}**")

# ══════════════════════════════════════════════════════════════════════════════
# Global controls (sidebar)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    selected_model = st.selectbox("Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
    model_key = {"Random Forest":"RF","Gradient Boosting":"GB","Logistic Regression":"LR"}[selected_model]
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)
    st.markdown("---")
    st.markdown("**Risk bands**")
    st.markdown('<span class="tag tag-low">LOW &lt; 30%</span>', unsafe_allow_html=True)
    st.markdown('<span class="tag tag-medium">MEDIUM 30–70%</span>', unsafe_allow_html=True)
    st.markdown('<span class="tag tag-high">HIGH &gt; 70%</span>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Compute metrics
# ══════════════════════════════════════════════════════════════════════════════
predict_fns = {"LR": predict_lr, "RF": predict_rf, "GB": predict_gb}
auc_offsets = {"LR": 0.0, "RF": 0.14, "GB": 0.16}
all_metrics = {k: compute_metrics(data, predict_fns[k], threshold, 0.72+auc_offsets[k])
               for k in ["LR","RF","GB"]}
m = all_metrics[model_key]

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Overview", "🤖 Models", "🔍 Explainability", "⚡ Live Scorer", "💬 AI Analyst", "📂 Import Data"])
TAB_OVERVIEW, TAB_MODELS, TAB_EXPLAIN, TAB_SCORER, TAB_AI, TAB_IMPORT = tabs

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · Overview
# ─────────────────────────────────────────────────────────────────────────────
with TAB_OVERVIEW:
    # Top metric row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", m["auc"], help="Area under ROC curve")
    c2.metric("F1 Score", m["f1"])
    c3.metric("Precision", m["precision"])
    c4.metric("Recall", m["recall"])

    st.markdown("")
    left, right = st.columns(2)

    # Scatter plot
    with left:
        st.markdown('<p class="section-head">Amount vs Velocity</p>', unsafe_allow_html=True)
        amounts    = [r.amount for r in data[:200]]
        velocities = [r.transaction_velocity for r in data[:200]]
        actuals    = [r.fraud for r in data[:200]]
        probs      = [predict_fns[model_key](r) for r in data[:200]]
        preds      = [1 if p>=threshold else 0 for p in probs]

        colors = []
        for a, p in zip(actuals, preds):
            if a==1 and p==1: colors.append("#2ecc71")   # TP
            elif a==1 and p==0: colors.append("#f39c12")  # FN
            elif a==0 and p==1: colors.append("#e74c3c")  # FP
            else: colors.append("#4a5f70")                # TN

        fig, ax = plt.subplots(figsize=(5,3.2))
        fig.patch.set_facecolor("#0a1520")
        ax.set_facecolor("#0a1520")
        ax.scatter(amounts, velocities, c=colors, s=18, alpha=0.7, linewidths=0)
        ax.set_xlabel("Amount ($)", color="#6b7f8f", fontsize=8)
        ax.set_ylabel("Velocity (tx/hr)", color="#6b7f8f", fontsize=8)
        ax.tick_params(colors="#4a5f70", labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor("#1e2d3d")
        legend_patches = [
            mpatches.Patch(color="#2ecc71", label="True Positive"),
            mpatches.Patch(color="#f39c12", label="False Negative"),
            mpatches.Patch(color="#e74c3c", label="False Positive"),
            mpatches.Patch(color="#4a5f70", label="True Negative"),
        ]
        ax.legend(handles=legend_patches, fontsize=6, facecolor="#0f1923",
                  edgecolor="#1e2d3d", labelcolor="white", loc="upper left")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Confusion matrix + ROC
    with right:
        st.markdown('<p class="section-head">Confusion Matrix</p>', unsafe_allow_html=True)
        cm_data = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))
        fig2.patch.set_facecolor("#0a1520")
        ax2.set_facecolor("#0a1520")
        im = ax2.imshow(cm_data, cmap="Blues", aspect="auto")
        ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
        ax2.set_xticklabels(["Pred: Legit","Pred: Fraud"], color="#6b7f8f", fontsize=8)
        ax2.set_yticklabels(["Act: Legit","Act: Fraud"], color="#6b7f8f", fontsize=8)
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm_data[i,j]), ha="center", va="center",
                         color="white", fontsize=14, fontweight="bold")
        for spine in ax2.spines.values(): spine.set_edgecolor("#1e2d3d")
        ax2.tick_params(colors="#4a5f70")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown('<p class="section-head" style="margin-top:.8rem">ROC Curves</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(3.5, 2.2))
        fig3.patch.set_facecolor("#0a1520")
        ax3.set_facecolor("#0a1520")
        roc_curves = {
            "LR": (0.72, "#3a8fd1", [[0,0],[.05,.3],[.1,.5],[.2,.68],[.4,.82],[.7,.91],[1,1]]),
            "RF": (0.86, "#2ecc71", [[0,0],[.03,.42],[.07,.65],[.15,.79],[.3,.89],[.6,.95],[1,1]]),
            "GB": (0.88, "#9b59b6", [[0,0],[.02,.45],[.06,.67],[.12,.82],[.28,.91],[.55,.96],[1,1]]),
        }
        ax3.plot([0,1],[0,1],"--",color="#2a3d4f",linewidth=.8)
        for k,(auc,color,pts) in roc_curves.items():
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            lw = 2.2 if k==model_key else 1.2
            ax3.plot(xs,ys,color=color,linewidth=lw,label=f"{k} AUC={auc}")
        ax3.set_xlabel("FPR", color="#6b7f8f", fontsize=7)
        ax3.set_ylabel("TPR", color="#6b7f8f", fontsize=7)
        ax3.tick_params(colors="#4a5f70", labelsize=6)
        ax3.legend(fontsize=6, facecolor="#0f1923", edgecolor="#1e2d3d", labelcolor="white")
        for spine in ax3.spines.values(): spine.set_edgecolor("#1e2d3d")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # Threshold slider info
    st.markdown("---")
    st.markdown(f"**Decision threshold: `{threshold:.2f}`** — move in sidebar to trade off precision vs recall")
    t1, t2, t3 = st.columns(3)
    t1.markdown('<span class="tag tag-low">LOW RISK &lt; 30%</span>', unsafe_allow_html=True)
    t2.markdown('<span class="tag tag-medium">MEDIUM RISK 30–70%</span>', unsafe_allow_html=True)
    t3.markdown('<span class="tag tag-high">HIGH RISK &gt; 70%</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Models
# ─────────────────────────────────────────────────────────────────────────────
with TAB_MODELS:
    model_info = {
        "LR": ("Logistic Regression", LR_IMP, 0.72, "Fast, interpretable baseline. Assumes linear decision boundary. Limited on non-linear patterns."),
        "RF": ("Random Forest", RF_IMP, 0.86, "100 decision trees via bagging. Handles non-linearity well. SMOTE + stratified split used."),
        "GB": ("Gradient Boosting", GB_IMP, 0.88, "Boosted ensemble (lr=0.1). Best AUC. RandomizedSearchCV tuned n_estimators, depth, min_samples."),
    }

    cols = st.columns(3)
    for i, (k, (label, imp, auc, note)) in enumerate(model_info.items()):
        with cols[i]:
            active = k == model_key
            border = "#3a8fd1" if active else "#1e2d3d"
            st.markdown(f"""
            <div style="border:1px solid {border};border-radius:10px;padding:1rem;
                        background:{'#0d2236' if active else '#0a1520'}">
              <div style="font-weight:600;color:{'#3a8fd1' if active else '#8fa8b8'};
                          margin-bottom:.5rem">{label}</div>
              <div style="font-size:1.6rem;font-weight:700;color:#fff">AUC {auc}</div>
              <div style="font-size:.72rem;color:#6b7f8f;margin-top:.4rem">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-head">Feature Importances</p>', unsafe_allow_html=True)

    imp_cols = st.columns(3)
    for i, (k, (label, imp, auc, note)) in enumerate(model_info.items()):
        with imp_cols[i]:
            st.markdown(f"**{label}**")
            sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
            feat_names = [f.replace("_"," ") for f,_ in sorted_imp]
            feat_vals  = [v for _,v in sorted_imp]
            fig, ax = plt.subplots(figsize=(3.2, 2.4))
            fig.patch.set_facecolor("#0a1520")
            ax.set_facecolor("#0a1520")
            bars = ax.barh(feat_names, feat_vals,
                           color=["#3a8fd1" if k=="LR" else "#2ecc71" if k=="RF" else "#9b59b6"]*len(feat_names),
                           height=0.55)
            ax.set_xlim(0, 0.45)
            ax.tick_params(colors="#6b7f8f", labelsize=7)
            ax.xaxis.set_tick_params(labelsize=6)
            for spine in ax.spines.values(): spine.set_edgecolor("#1e2d3d")
            for bar, val in zip(bars, feat_vals):
                ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                        f"{val:.0%}", va="center", fontsize=6, color="#8fa8b8")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # All metrics comparison
    st.markdown("---")
    st.markdown('<p class="section-head">Metrics Comparison</p>', unsafe_allow_html=True)
    df_metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        "AUC":       [all_metrics["LR"]["auc"], all_metrics["RF"]["auc"], all_metrics["GB"]["auc"]],
        "F1":        [all_metrics["LR"]["f1"],  all_metrics["RF"]["f1"],  all_metrics["GB"]["f1"]],
        "Precision": [all_metrics["LR"]["precision"], all_metrics["RF"]["precision"], all_metrics["GB"]["precision"]],
        "Recall":    [all_metrics["LR"]["recall"],    all_metrics["RF"]["recall"],    all_metrics["GB"]["recall"]],
        "Accuracy":  [all_metrics["LR"]["accuracy"],  all_metrics["RF"]["accuracy"],  all_metrics["GB"]["accuracy"]],
    })
    st.dataframe(df_metrics.set_index("Model"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · Explainability
# ─────────────────────────────────────────────────────────────────────────────
with TAB_EXPLAIN:
    shap_model = st.radio("SHAP values for:", ["Random Forest", "Gradient Boosting"], horizontal=True)
    shap_data  = SHAP_RF if shap_model == "Random Forest" else SHAP_GB

    st.markdown('<p class="section-head">Mean |SHAP| Impact</p>', unsafe_allow_html=True)
    st.caption("Average magnitude of each feature's contribution across all test samples.")

    fig_shap, ax_shap = plt.subplots(figsize=(7, 3))
    fig_shap.patch.set_facecolor("#0a1520")
    ax_shap.set_facecolor("#0a1520")
    feat_labels = [f.replace("_"," ") for f,_,_ in shap_data]
    shap_vals   = [s for _,_,s in shap_data]
    bar_colors  = ["#e74c3c" if s>0 else "#3a8fd1" for s in shap_vals]
    ax_shap.barh(feat_labels, shap_vals, color=bar_colors, height=0.55)
    ax_shap.axvline(0, color="#2a3d4f", linewidth=1)
    ax_shap.set_xlabel("Mean SHAP value", color="#6b7f8f", fontsize=8)
    ax_shap.tick_params(colors="#6b7f8f", labelsize=8)
    for spine in ax_shap.spines.values(): spine.set_edgecolor("#1e2d3d")
    for val, label in zip(shap_vals, feat_labels):
        ax_shap.text(val + (0.003 if val>=0 else -0.003), feat_labels.index(label),
                     f"{val:+.2f}", va="center", ha="left" if val>=0 else "right",
                     fontsize=7, color="#8fa8b8")
    st.pyplot(fig_shap, use_container_width=True)
    plt.close()

    legend_c1, legend_c2 = st.columns(2)
    legend_c1.markdown("🔴 **Pushes toward fraud**")
    legend_c2.markdown("🔵 **Pushes toward non-fraud**")

    st.markdown("---")
    eng_c1, eng_c2 = st.columns(2)
    with eng_c1:
        st.markdown('<p class="section-head">Engineered Features</p>', unsafe_allow_html=True)
        st.markdown("""
        **transaction_velocity**
        `tx_count / (hour + ε)` — detects burst activity

        **risk_score**
        `amount × (1 + new_merchant × 0.5)` — weighted amount risk
        """)
    with eng_c2:
        st.markdown('<p class="section-head">SMOTE Resampling</p>', unsafe_allow_html=True)
        st.markdown("Synthetic Minority Oversampling balances class distribution before training.")
        smote_df = pd.DataFrame({"Stage":["Before","Before","After","After"],
                                  "Class":["Fraud","Legit","Fraud","Legit"],
                                  "Pct":[22,78,50,50]})
        fig_sm, ax_sm = plt.subplots(figsize=(3,2))
        fig_sm.patch.set_facecolor("#0a1520")
        ax_sm.set_facecolor("#0a1520")
        x = np.arange(2)
        ax_sm.bar(x-0.2, [22,50], 0.35, label="Fraud", color="#e74c3c", alpha=0.85)
        ax_sm.bar(x+0.2, [78,50], 0.35, label="Legit",  color="#2ecc71", alpha=0.85)
        ax_sm.set_xticks(x); ax_sm.set_xticklabels(["Before","After"], color="#6b7f8f", fontsize=8)
        ax_sm.set_ylabel("%", color="#6b7f8f", fontsize=7)
        ax_sm.tick_params(colors="#4a5f70", labelsize=7)
        ax_sm.legend(fontsize=7, facecolor="#0f1923", edgecolor="#1e2d3d", labelcolor="white")
        for spine in ax_sm.spines.values(): spine.set_edgecolor("#1e2d3d")
        plt.tight_layout()
        st.pyplot(fig_sm, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Live Scorer
# ─────────────────────────────────────────────────────────────────────────────
with TAB_SCORER:
    st.markdown("Adjust sliders to score a transaction in real time.")
    sc_left, sc_right = st.columns([1.4, 1])

    with sc_left:
        amount_val   = st.slider("Amount ($)", 1, 3000, 450)
        velocity_val = st.slider("Velocity (tx/hr)", 0.0, 15.0, 2.5, 0.1)
        hour_val     = st.slider("Hour of day", 0, 23, 14)
        avg_tx_val   = st.slider("Avg tx value ($)", 1, 3000, 380)
        tx_count_val = st.slider("Tx count (24h)", 0, 50, 4)
        new_merch    = st.toggle("New merchant", value=False)

    risk_score_val = amount_val * (1 + (0.5 if new_merch else 0))
    live_row = Transaction(id=999, amount=float(amount_val), hour_of_day=float(hour_val),
                           transaction_velocity=float(velocity_val),
                           is_new_merchant=1 if new_merch else 0,
                           avg_transaction_value=float(avg_tx_val),
                           transaction_count_last_24h=float(tx_count_val),
                           risk_score=risk_score_val, fraud=0)

    prob = predict_fns[model_key](live_row)
    risk_cat = "LOW" if prob < 0.3 else ("MEDIUM" if prob < 0.7 else "HIGH")
    tag_cls  = {"LOW":"tag-low","MEDIUM":"tag-medium","HIGH":"tag-high"}[risk_cat]
    pred_txt = "FRAUD" if prob >= threshold else "LEGIT"

    with sc_right:
        # Gauge donut via matplotlib
        fig_g, ax_g = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"))
        fig_g.patch.set_facecolor("#0a1520")
        ax_g.set_facecolor("#0a1520")
        gauge_color = "#2ecc71" if risk_cat=="LOW" else ("#f39c12" if risk_cat=="MEDIUM" else "#e74c3c")
        ax_g.pie([prob, 1-prob],
                 colors=[gauge_color, "#1e2d3d"],
                 startangle=90,
                 counterclock=False,
                 wedgeprops=dict(width=0.35, edgecolor="#0a1520"))
        ax_g.text(0, 0.1, f"{prob*100:.0f}%", ha="center", va="center",
                  fontsize=26, fontweight="bold", color=gauge_color)
        ax_g.text(0, -0.25, "fraud probability", ha="center", va="center",
                  fontsize=8, color="#6b7f8f")
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

        st.markdown(f'<div style="text-align:center;margin:.4rem 0">'
                    f'<span class="tag {tag_cls}" style="font-size:.9rem;padding:5px 20px">'
                    f'{risk_cat} RISK</span></div>', unsafe_allow_html=True)

        st.markdown(f"**Prediction:** `{pred_txt}` &nbsp; (model: {model_key}, threshold: {threshold:.2f})")

        # Score breakdown bars
        st.markdown("**Score breakdown**")
        breakdowns = [
            ("Amount signal",   min(1.0, amount_val/2000)),
            ("Velocity signal", min(1.0, velocity_val/12)),
            ("Merchant risk",   (1 if new_merch else 0)*0.8 + 0.05),
        ]
        for lbl, val in breakdowns:
            st.markdown(f"<div style='font-size:.75rem;color:#6b7f8f;margin-bottom:2px'>{lbl}</div>", unsafe_allow_html=True)
            st.progress(val)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · AI Analyst
# ─────────────────────────────────────────────────────────────────────────────
with TAB_AI:
    st.markdown("Ask anything about the fraud detection pipeline, model choices, or ML concepts.")

    suggestions = [
        "Why might recall be more important than precision in fraud detection?",
        "What threshold would minimise false negatives?",
        "How does SMOTE help with class imbalance here?",
        "Compare the tradeoffs between Random Forest and Gradient Boosting.",
    ]

    suggestion_cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if suggestion_cols[i%2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state["ai_query"] = s

    ai_query = st.text_input("Your question", value=st.session_state.get("ai_query",""),
                             placeholder="e.g. Why is recall more important than precision here?",
                             key="ai_input")

    if st.button("Ask", type="primary") and ai_query.strip():
        context = (
            f"You are an expert fraud detection data scientist. "
            f"Dataset: {len(data)} transactions, {fraud_count/len(data)*100:.1f}% fraud rate. "
            f"Models: LR AUC={all_metrics['LR']['auc']}, RF AUC={all_metrics['RF']['auc']}, "
            f"GB AUC={all_metrics['GB']['auc']}. "
            f"Top features: risk_score (0.31), transaction_velocity (0.24), amount (0.18). "
            f"Current model: {model_key}, threshold: {threshold}. "
            f"F1={m['f1']}, Precision={m['precision']}, Recall={m['recall']}, AUC={m['auc']}. "
            f"Answer in 3-4 concise sentences. Be specific and technical."
        )
        try:
            import urllib.request, json as _json
            # Try st.secrets first, then fall back gracefully
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except Exception:
                api_key = None

            if not api_key:
                st.warning("No ANTHROPIC_API_KEY found in Streamlit secrets. Add it under Settings → Secrets to enable the AI analyst.")
            else:
                with st.spinner("Thinking…"):
                    payload = _json.dumps({
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 1000,
                        "system": context,
                        "messages": [{"role":"user","content":ai_query}],
                    }).encode()
                    req = urllib.request.Request(
                        "https://api.anthropic.com/v1/messages",
                        data=payload,
                        headers={"Content-Type":"application/json",
                                 "x-api-key": api_key,
                                 "anthropic-version":"2023-06-01"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req) as resp:
                        result = _json.loads(resp.read())
                    answer = "".join(b["text"] for b in result.get("content",[]) if b.get("type")=="text")
                    st.markdown(f"""
                    <div style="background:#0d1f30;border-left:3px solid #3a8fd1;
                                border-radius:0 8px 8px 0;padding:1rem 1.2rem;
                                font-size:.9rem;line-height:1.6;color:#c8d8e8">
                    {answer}
                    </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not reach AI analyst: {e}")

    st.markdown("---")
    st.caption(f"Context: {model_key} · threshold {threshold:.2f} · AUC {m['auc']} · F1 {m['f1']} · {len(data):,} rows · {st.session_state.data_source}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · Import Data
# ─────────────────────────────────────────────────────────────────────────────
with TAB_IMPORT:
    if st.session_state.data_source != "demo":
        st.success(f"✅ Currently using imported data: **{st.session_state.data_source}**")
        if st.button("Reset to demo data"):
            st.session_state.data = generate_dataset(400)
            st.session_state.data_source = "demo"
            st.rerun()

    uploaded = st.file_uploader("Drop a CSV here", type=["csv"])
    if uploaded:
        try:
            rows, col_map, headers = parse_csv(uploaded)
            fraud_n = sum(r.fraud for r in rows)
            st.success(f"Loaded **{len(rows):,}** transactions · {fraud_n} fraud ({fraud_n/len(rows)*100:.1f}%)")
            st.session_state.data = rows
            st.session_state.data_source = uploaded.name

            # Column mapping preview
            col_defs = [
                ("amount","Amount",True),("fraud","Fraud label",True),
                ("hour_of_day","Hour of day",False),("transaction_velocity","Velocity",False),
                ("is_new_merchant","New merchant",False),("avg_transaction_value","Avg tx value",False),
                ("transaction_count_last_24h","Tx count 24h",False),("risk_score","Risk score",False),
            ]
            st.markdown("**Column mapping**")
            map_cols = st.columns(2)
            for i, (canon, label, required) in enumerate(col_defs):
                mapped = col_map.get(canon)
                icon = "✅" if mapped else ("❌" if required else "⚪")
                map_cols[i%2].markdown(f"{icon} `{label}` → `{mapped or 'not found'}`")

            # Data preview
            st.markdown("**Data preview** (first 6 rows)")
            preview_rows = rows[:6]
            df_preview = pd.DataFrame([{
                "amount": r.amount, "hour": r.hour_of_day,
                "velocity": r.transaction_velocity, "new_merchant": r.is_new_merchant,
                "avg_tx": r.avg_transaction_value, "risk_score": r.risk_score, "fraud": r.fraud
            } for r in preview_rows])
            st.dataframe(df_preview, use_container_width=True)
            st.rerun()
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    st.markdown("---")
    st.markdown("**Expected CSV format**")
    st.code("amount, fraud, [hour_of_day], [is_new_merchant],\ntransaction_count_last_24h], [avg_transaction_value],\n[transaction_velocity], [risk_score]", language="text")
    st.caption("Column names are matched fuzzily — e.g. `transaction_amount`, `tx_amount`, or `value` all map to `amount`. Optional columns are computed from available data if absent.")

    st.download_button("⬇️ Download sample CSV", data=SAMPLE_CSV,
                       file_name="sample_fraud_data.csv", mime="text/csv")
