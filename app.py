import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud & Campaign Analytics",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .metric-card {
        background: #0f1117;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #e2e8f0;
        line-height: 1;
    }
    .metric-value.good { color: #34d399; }
    .metric-value.warn { color: #fbbf24; }
    .metric-value.bad  { color: #f87171; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #2a2d3a;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    div[data-testid="stSidebar"] { background: #0a0b0f; }
    .stSlider label { font-size: 0.8rem; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    predictions = pd.read_csv("predictions_output.csv")
    rules = pd.read_csv("association_rules.csv")
    with open("metrics_summary.json") as f:
        metrics = json.load(f)

    # Clean frozenset strings for display
    def clean_set(s):
        return s.replace("frozenset({", "").replace("})", "").replace("'", "").strip()

    rules["antecedents_str"] = rules["antecedents"].apply(clean_set)
    rules["consequents_str"] = rules["consequents"].apply(clean_set)
    return predictions, rules, metrics

predictions_df, rules_df, metrics = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Controls")
    st.markdown("---")

    threshold = st.slider(
        "Prediction threshold",
        min_value=0.0, max_value=1.0,
        value=float(metrics["optimal_threshold"]),
        step=0.01,
        help="Adjust the classification cutoff. Default is the model's optimal threshold."
    )

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown(f"`{metrics['model']}`")
    st.markdown(f"Depth: `{metrics['best_params']['max_depth']}`  \nEstimators: `{metrics['best_params']['n_estimators']}`")

# ── Apply threshold ───────────────────────────────────────────────────────────
predictions_df["Dynamic_Pred"] = (
    predictions_df["Predicted_Probability"] >= threshold
).astype(int)

y_true = predictions_df["Actual_Label"]
y_pred = predictions_df["Dynamic_Pred"]
y_prob = predictions_df["Predicted_Probability"]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, zero_division=0)
rec   = recall_score(y_true, y_pred, zero_division=0)
f1    = f1_score(y_true, y_pred, zero_division=0)
spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
roc_auc_val = auc(fpr_c, tpr_c)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Fraud & Campaign Analytics")
st.markdown(
    f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#6b7280;'>"
    f"Threshold: <b style='color:#6366f1'>{threshold:.2f}</b> &nbsp;·&nbsp; "
    f"{len(predictions_df)} customers &nbsp;·&nbsp; {metrics['model']}"
    f"</span>",
    unsafe_allow_html=True
)

# ── KPI row ───────────────────────────────────────────────────────────────────
def color_class(val, good=0.7, warn=0.5):
    if val >= good: return "good"
    if val >= warn: return "warn"
    return "bad"

def kpi(label, value, fmt=".1%", extra_class=""):
    cls = extra_class or color_class(value)
    display = format(value, fmt) if isinstance(value, float) else str(value)
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{display}</div>
    </div>"""

cols = st.columns(6)
kpis = [
    ("Accuracy",    acc,         ".1%", ""),
    ("Precision",   prec,        ".1%", ""),
    ("Recall",      rec,         ".1%", ""),
    ("F1 Score",    f1,          ".1%", ""),
    ("Specificity", spec,        ".1%", ""),
    ("ROC AUC",     roc_auc_val, ".3f", ""),
]
for col, (label, val, fmt, cls) in zip(cols, kpis):
    with col:
        st.markdown(kpi(label, val, fmt, cls), unsafe_allow_html=True)

# ── Confusion matrix + ROC ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    data = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    colors = [["#1e3a5f", "#5b1d1d"], ["#5b1d1d", "#1a4731"]]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=colors[i][j]))
            ax.text(j+0.5, 1.5-i, f"{labels[i][j]}\n{data[i][j]}",
                    ha='center', va='center', fontsize=13,
                    fontfamily='monospace', color='#e2e8f0', fontweight='bold')

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"], color="#9ca3af", fontsize=8)
    ax.set_yticklabels(["Actual 1", "Actual 0"], color="#9ca3af", fontsize=8)
    ax.tick_params(colors='#9ca3af')
    for spine in ax.spines.values(): spine.set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col2:
    st.markdown("**ROC Curve**")
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.plot(fpr_c, tpr_c, color="#6366f1", lw=2, label=f"AUC = {roc_auc_val:.3f}")
    ax.plot([0,1],[0,1], color="#374151", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="#9ca3af", fontsize=8)
    ax.set_ylabel("True Positive Rate", color="#9ca3af", fontsize=8)
    ax.tick_params(colors="#6b7280", labelsize=7)
    ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e2e8f0")
    for spine in ax.spines.values(): spine.set_color("#2a2d3a")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col3:
    st.markdown("**Precision & Recall vs Threshold**")
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.plot(pr_thresholds, precisions[:-1], color="#34d399", lw=2, label="Precision")
    ax.plot(pr_thresholds, recalls[:-1], color="#fbbf24", lw=2, label="Recall")
    ax.axvline(x=threshold, color="#6366f1", linestyle="--", lw=1.2, label=f"t={threshold:.2f}")
    ax.set_xlabel("Threshold", color="#9ca3af", fontsize=8)
    ax.tick_params(colors="#6b7280", labelsize=7)
    ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e2e8f0")
    for spine in ax.spines.values(): spine.set_color("#2a2d3a")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Model comparison ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

comparison = metrics["all_model_comparison"]
comp_df = pd.DataFrame(comparison).T[["accuracy","precision","recall","f1_score","auc","balanced_accuracy"]]
comp_df.columns = ["Accuracy","Precision","Recall","F1","AUC","Balanced Acc"]
comp_df = comp_df.round(3)

st.dataframe(
    comp_df.style
        .background_gradient(cmap="RdYlGn", axis=None, vmin=0.4, vmax=0.8)
        .format("{:.3f}"),
    use_container_width=True
)

# ── Campaign & segment insights ───────────────────────────────────────────────
st.markdown('<div class="section-header">Campaign & Segment Insights</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Response Rate by Segment**")
    resp = predictions_df.groupby("Segment")["Campaign_Response"].mean().sort_values(ascending=True) * 100
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    bars = ax.barh(resp.index, resp.values, color=["#6366f1","#818cf8","#a5b4fc"][:len(resp)])
    ax.set_xlabel("Response Rate (%)", color="#9ca3af", fontsize=8)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for bar, val in zip(bars, resp.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', color="#e2e8f0", fontsize=8)
    for spine in ax.spines.values(): spine.set_color("#2a2d3a")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col2:
    st.markdown("**Recommendation Category Distribution**")
    rec_counts = predictions_df["Recommendation_Category"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    palette = ["#6366f1","#34d399","#fbbf24","#f87171","#a78bfa"]
    wedges, texts, autotexts = ax.pie(
        rec_counts.values,
        labels=rec_counts.index,
        autopct="%1.0f%%",
        colors=palette[:len(rec_counts)],
        textprops={"color":"#e2e8f0","fontsize":8},
        wedgeprops={"linewidth":1.5,"edgecolor":"#0f1117"}
    )
    for at in autotexts: at.set_color("#0f1117"); at.set_fontweight("bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Association rules ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Association Rules</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.05, 0.01)
with col2:
    min_lift = st.slider("Min lift", 0.0, float(rules_df["lift"].max()), 0.5, 0.05)
with col3:
    top_n = st.slider("Show top N rules", 5, 25, 10)

filtered = rules_df[
    (rules_df["confidence"] >= min_conf) &
    (rules_df["lift"] >= min_lift)
].sort_values("lift", ascending=False).head(top_n)

display_cols = ["antecedents_str","consequents_str","confidence","lift","support"]
display_labels = {"antecedents_str":"Antecedents","consequents_str":"Consequents",
                  "confidence":"Confidence","lift":"Lift","support":"Support"}

st.dataframe(
    filtered[display_cols].rename(columns=display_labels)
        .style.format({"Confidence":"{:.3f}","Lift":"{:.3f}","Support":"{:.4f}"})
        .background_gradient(subset=["Lift"], cmap="Blues"),
    use_container_width=True,
    height=300
)

# ── Customer explorer ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Customer Explorer</div>', unsafe_allow_html=True)

seg_filter = st.multiselect(
    "Filter by segment",
    options=predictions_df["Segment"].unique().tolist(),
    default=predictions_df["Segment"].unique().tolist()
)

show_df = predictions_df[predictions_df["Segment"].isin(seg_filter)].copy()
show_df["Correct"] = show_df["Actual_Label"] == show_df["Dynamic_Pred"]

st.dataframe(
    show_df[["Customer_ID","Actual_Label","Dynamic_Pred","Predicted_Probability",
             "Segment","Campaign_Response","Recommendation_Category","Correct"]]
        .rename(columns={"Dynamic_Pred":"Predicted","Predicted_Probability":"Probability"})
        .style.format({"Probability":"{:.3f}"})
        .applymap(lambda v: "color: #34d399" if v is True else ("color: #f87171" if v is False else ""),
                  subset=["Correct"]),
    use_container_width=True,
    height=350
)
