import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation & Recommendation Engine",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .metric-card {
        background: #0f1117; border: 1px solid #2a2d3a;
        border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 0.5rem;
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
        font-weight: 600; color: #e2e8f0; line-height: 1;
    }
    .metric-value.good { color: #34d399; }
    .metric-value.warn { color: #fbbf24; }
    .metric-value.bad  { color: #f87171; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #6366f1;
        text-transform: uppercase; letter-spacing: 0.12em;
        border-bottom: 1px solid #2a2d3a; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0;
    }
    .byline {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: #4b5563; margin-top: 0.2rem; margin-bottom: 1.5rem;
    }
    .upload-box {
        background: #0f1117; border: 2px dashed #2a2d3a; border-radius: 12px;
        padding: 2rem; text-align: center; margin-bottom: 1.5rem;
    }
    .mode-badge {
        display: inline-block; font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem; padding: 0.2rem 0.6rem; border-radius: 999px;
        text-transform: uppercase; letter-spacing: 0.1em; margin-left: 0.5rem;
    }
    .mode-default { background: #1e3a5f; color: #93c5fd; }
    .mode-upload  { background: #1a4731; color: #6ee7b7; }
    div[data-testid="stSidebar"] { background: #0a0b0f; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        background: #0f1117; border: 1px solid #2a2d3a; border-radius: 6px;
        color: #6b7280; padding: 0.4rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e1b4b !important; border-color: #6366f1 !important; color: #a5b4fc !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def color_class(val):
    if val >= 0.7: return "good"
    if val >= 0.5: return "warn"
    return "bad"

def kpi(label, value, fmt=".1%"):
    cls = color_class(value)
    display = format(value, fmt)
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{display}</div>
    </div>"""

def style_correct(val):
    if val is True:   return "color: #34d399"
    if val is False:  return "color: #f87171"
    return ""

def dark_fig(figsize=(4, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    return fig, ax

def style_ax(ax):
    ax.tick_params(colors="#6b7280", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#2a2d3a")

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
    return {
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "spec": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "fpr":  fpr_c, "tpr": tpr_c,
        "roc_auc": auc(fpr_c, tpr_c),
    }

# ── Load default data ─────────────────────────────────────────────────────────
@st.cache_data
def load_default():
    predictions = pd.read_csv("predictions_output.csv")
    rules = pd.read_csv("association_rules.csv")
    with open("metrics_summary.json") as f:
        metrics = json.load(f)
    def clean_set(s):
        return s.replace("frozenset({", "").replace("})", "").replace("'", "").strip()
    rules["antecedents_str"] = rules["antecedents"].apply(clean_set)
    rules["consequents_str"] = rules["consequents"].apply(clean_set)
    return predictions, rules, metrics

default_predictions, default_rules, default_metrics = load_default()

# ── Train models on uploaded data ────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression":   LogisticRegression(max_iter=1000),
    "Random Forest":         RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                   SVC(probability=True, random_state=42),
    "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=5),
}
MODEL_COLORS = {
    "Logistic Regression":  "#6366f1",
    "Random Forest":        "#34d399",
    "Gradient Boosting":    "#fbbf24",
    "SVM":                  "#f87171",
    "K-Nearest Neighbors":  "#a78bfa",
}

@st.cache_data
def train_on_upload(csv_bytes, target_col, selected_models):
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    if target_col not in df.columns:
        return None, None, "Target column not found."

    # Encode categoricals
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    df_enc = df_enc.dropna()

    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]

    if y.nunique() != 2:
        return None, None, "Target column must be binary (2 classes)."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}
    for name in selected_models:
        clf = MODEL_REGISTRY[name]
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        results[name] = {"y_pred": y_pred, "y_prob": y_prob}

    test_df = X_test.copy()
    test_df["Actual_Label"] = y_test.values
    return results, test_df, None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Controls")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    data_mode = st.radio("", ["Use default dataset", "Upload my own CSV"], label_visibility="collapsed")

    uploaded_file  = None
    target_col     = None
    selected_models = list(MODEL_REGISTRY.keys())[:3]
    upload_error   = None
    upload_results = None
    upload_test_df = None

    if data_mode == "Upload my own CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            target_col = st.selectbox("Target (label) column", preview_df.columns.tolist())
            selected_models = st.multiselect(
                "Models to train",
                list(MODEL_REGISTRY.keys()),
                default=list(MODEL_REGISTRY.keys())[:3]
            )
            if st.button("🚀 Train models", use_container_width=True):
                with st.spinner("Training…"):
                    upload_results, upload_test_df, upload_error = train_on_upload(
                        uploaded_file.read(), target_col, selected_models
                    )
                if upload_results:
                    st.session_state["upload_results"] = upload_results
                    st.session_state["upload_test_df"] = upload_test_df
                    st.session_state["upload_target"]  = target_col
                    st.success("Done!")
                else:
                    st.error(upload_error or "Training failed.")

    st.markdown("---")
    st.markdown("### ⚙️ Threshold")
    threshold = st.slider(
        "Prediction threshold",
        min_value=0.0, max_value=1.0,
        value=float(default_metrics["optimal_threshold"]),
        step=0.01,
        help="Applies to the default dataset view."
    )

    if data_mode == "Use default dataset":
        st.markdown("---")
        st.markdown("**Default Model**")
        st.markdown(f"`{default_metrics['model']}`")
        st.markdown(
            f"Depth: `{default_metrics['best_params']['max_depth']}`  \n"
            f"Estimators: `{default_metrics['best_params']['n_estimators']}`"
        )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
mode_badge = (
    '<span class="mode-badge mode-upload">uploaded data</span>'
    if data_mode == "Upload my own CSV" and "upload_results" in st.session_state
    else '<span class="mode-badge mode-default">default data</span>'
)

st.markdown(f"# Customer Segmentation & Recommendation Engine {mode_badge}", unsafe_allow_html=True)
st.markdown(
    "<div class='byline'>By: Ashwath Shankarkri · Soham Srinivas · Keshauv Prakash · Sai Vasanth</div>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_models, tab_campaign, tab_rules, tab_explorer, tab_export = st.tabs([
    "📊 Overview", "🤖 Model Lab", "📣 Campaign", "🔗 Rules", "🔎 Explorer", "⬇ Export"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW (default dataset, threshold-driven)
# ──────────────────────────────────────────────────────────────────────────────
with tab_overview:
    default_predictions["Dynamic_Pred"] = (
        default_predictions["Predicted_Probability"] >= threshold
    ).astype(int)

    y_true = default_predictions["Actual_Label"]
    y_pred = default_predictions["Dynamic_Pred"]
    y_prob = default_predictions["Predicted_Probability"]
    m = compute_metrics(y_true, y_pred, y_prob)

    st.markdown(
        f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#6b7280;'>"
        f"Threshold: <b style='color:#6366f1'>{threshold:.2f}</b> &nbsp;·&nbsp; "
        f"{len(default_predictions)} customers &nbsp;·&nbsp; {default_metrics['model']}"
        f"</span>", unsafe_allow_html=True
    )

    # KPIs
    cols = st.columns(6)
    for col, (label, val, fmt) in zip(cols, [
        ("Accuracy", m["acc"], ".1%"), ("Precision", m["prec"], ".1%"),
        ("Recall",   m["rec"], ".1%"), ("F1 Score",  m["f1"],  ".1%"),
        ("Specificity", m["spec"], ".1%"), ("ROC AUC", m["roc_auc"], ".3f"),
    ]):
        with col: st.markdown(kpi(label, val, fmt), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Confusion Matrix**")
        fig, ax = dark_fig()
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        data_cm = np.array([[tn, fp],[fn, tp]])
        lbls   = [["TN","FP"],["FN","TP"]]
        clrs   = [["#1e3a5f","#5b1d1d"],["#5b1d1d","#1a4731"]]
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j,1-i),1,1,color=clrs[i][j]))
                ax.text(j+.5,1.5-i,f"{lbls[i][j]}\n{data_cm[i][j]}",
                        ha='center',va='center',fontsize=13,
                        fontfamily='monospace',color='#e2e8f0',fontweight='bold')
        ax.set_xlim(0,2); ax.set_ylim(0,2)
        ax.set_xticks([.5,1.5]); ax.set_yticks([.5,1.5])
        ax.set_xticklabels(["Pred 0","Pred 1"],color="#9ca3af",fontsize=8)
        ax.set_yticklabels(["Actual 1","Actual 0"],color="#9ca3af",fontsize=8)
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.markdown("**ROC Curve**")
        fig, ax = dark_fig()
        ax.plot(m["fpr"], m["tpr"], color="#6366f1", lw=2, label=f"AUC={m['roc_auc']:.3f}")
        ax.plot([0,1],[0,1], color="#374151", lw=1, linestyle="--")
        ax.set_xlabel("FPR", color="#9ca3af", fontsize=8)
        ax.set_ylabel("TPR", color="#9ca3af", fontsize=8)
        ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e2e8f0")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c3:
        st.markdown("**Precision & Recall vs Threshold**")
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
        fig, ax = dark_fig()
        ax.plot(pr_thresholds, precisions[:-1], color="#34d399", lw=2, label="Precision")
        ax.plot(pr_thresholds, recalls[:-1],    color="#fbbf24", lw=2, label="Recall")
        ax.axvline(x=threshold, color="#6366f1", linestyle="--", lw=1.2, label=f"t={threshold:.2f}")
        ax.set_xlabel("Threshold", color="#9ca3af", fontsize=8)
        ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e2e8f0")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Model comparison table from JSON
    st.markdown('<div class="section-header">Model Comparison (from training run)</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame(default_metrics["all_model_comparison"]).T[
        ["accuracy","precision","recall","f1_score","auc","balanced_accuracy"]
    ]
    comp_df.columns = ["Accuracy","Precision","Recall","F1","AUC","Balanced Acc"]
    st.dataframe(
        comp_df.round(3).style
            .background_gradient(cmap="RdYlGn", axis=None, vmin=0.4, vmax=0.8)
            .format("{:.3f}"),
        use_container_width=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL LAB (train multiple models on uploaded or default data)
# ──────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown('<div class="section-header">Model Lab</div>', unsafe_allow_html=True)

    has_upload = "upload_results" in st.session_state

    if not has_upload:
        # Train all models on default predictions using Predicted_Probability as the score
        # and re-train classifiers on available numeric features
        st.info("💡 Upload a CSV in the sidebar and click **Train models** to compare classifiers on your own data. Showing default dataset model comparison below.")

        # Show bar chart comparison of models from JSON
        comp = default_metrics["all_model_comparison"]
        metrics_to_plot = ["accuracy","f1_score","auc","balanced_accuracy"]
        model_names = list(comp.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        for i, (mname, mcolor) in enumerate(zip(model_names, ["#6366f1","#34d399","#fbbf24"])):
            vals = [comp[mname][k] for k in metrics_to_plot]
            ax.bar(x + i*width, vals, width, label=mname, color=mcolor, alpha=0.9)
        ax.set_xticks(x + width); ax.set_xticklabels(["Accuracy","F1","AUC","Balanced Acc"], color="#9ca3af", fontsize=9)
        ax.set_ylim(0, 1); ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e2e8f0")
        ax.set_ylabel("Score", color="#9ca3af", fontsize=8)
        style_ax(ax)
        st.pyplot(fig, use_container_width=True); plt.close()

    else:
        results    = st.session_state["upload_results"]
        test_df    = st.session_state["upload_test_df"]
        tgt        = st.session_state["upload_target"]
        y_true_up  = test_df["Actual_Label"]

        # ── Per-model KPIs ────────────────────────────────────────────────────
        st.markdown("#### Per-model metrics")
        summary_rows = []
        all_metrics  = {}
        for mname, res in results.items():
            met = compute_metrics(y_true_up, res["y_pred"], res["y_prob"])
            all_metrics[mname] = met
            summary_rows.append({
                "Model": mname,
                "Accuracy":  round(met["acc"],  3),
                "Precision": round(met["prec"], 3),
                "Recall":    round(met["rec"],  3),
                "F1":        round(met["f1"],   3),
                "AUC":       round(met["roc_auc"], 3),
                "Specificity": round(met["spec"], 3),
            })

        sum_df = pd.DataFrame(summary_rows).set_index("Model")
        st.dataframe(
            sum_df.style
                .background_gradient(cmap="RdYlGn", axis=None, vmin=0.4, vmax=0.9)
                .format("{:.3f}"),
            use_container_width=True
        )

        # ── ROC curves overlay ────────────────────────────────────────────────
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
        for mname, met in all_metrics.items():
            color = MODEL_COLORS.get(mname, "#e2e8f0")
            ax.plot(met["fpr"], met["tpr"], color=color, lw=2,
                    label=f"{mname} (AUC={met['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], color="#374151", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate", color="#9ca3af", fontsize=9)
        ax.set_ylabel("True Positive Rate",  color="#9ca3af", fontsize=9)
        ax.legend(fontsize=8, facecolor="#1f2937", labelcolor="#e2e8f0")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── Confusion matrices side by side ───────────────────────────────────
        st.markdown("#### Confusion Matrices")
        ncols = min(len(results), 3)
        cm_cols = st.columns(ncols)
        for idx, (mname, met) in enumerate(all_metrics.items()):
            with cm_cols[idx % ncols]:
                st.markdown(f"**{mname}**")
                fig, ax = dark_fig((3.5, 2.8))
                tn2,fp2,fn2,tp2 = met["tn"],met["fp"],met["fn"],met["tp"]
                d2 = np.array([[tn2,fp2],[fn2,tp2]])
                l2 = [["TN","FP"],["FN","TP"]]
                c2 = [["#1e3a5f","#5b1d1d"],["#5b1d1d","#1a4731"]]
                for i in range(2):
                    for j in range(2):
                        ax.add_patch(plt.Rectangle((j,1-i),1,1,color=c2[i][j]))
                        ax.text(j+.5,1.5-i,f"{l2[i][j]}\n{d2[i][j]}",
                                ha='center',va='center',fontsize=11,
                                fontfamily='monospace',color='#e2e8f0',fontweight='bold')
                ax.set_xlim(0,2); ax.set_ylim(0,2)
                ax.set_xticks([.5,1.5]); ax.set_yticks([.5,1.5])
                ax.set_xticklabels(["Pred 0","Pred 1"],color="#9ca3af",fontsize=7)
                ax.set_yticklabels(["Act 1","Act 0"],  color="#9ca3af",fontsize=7)
                ax.tick_params(colors='#9ca3af')
                for spine in ax.spines.values(): spine.set_visible(False)
                st.pyplot(fig, use_container_width=True); plt.close()

        # ── Best model highlight ──────────────────────────────────────────────
        best = sum_df["F1"].idxmax()
        st.success(f"🏆 Best model by F1: **{best}** — F1 = {sum_df.loc[best,'F1']:.3f}, AUC = {sum_df.loc[best,'AUC']:.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — CAMPAIGN & SEGMENT INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
with tab_campaign:
    st.markdown('<div class="section-header">Campaign & Segment Insights</div>', unsafe_allow_html=True)

    # Use uploaded data if available and has the right columns, else default
    if (
        "upload_results" in st.session_state and
        "Campaign_Response" in st.session_state["upload_test_df"].columns and
        "Segment" in st.session_state["upload_test_df"].columns
    ):
        camp_df = st.session_state["upload_test_df"]
    else:
        camp_df = default_predictions

    if "Campaign_Response" in camp_df.columns and "Segment" in camp_df.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Response Rate by Segment**")
            resp = camp_df.groupby("Segment")["Campaign_Response"].mean().sort_values(ascending=True) * 100
            fig, ax = dark_fig((5,3))
            bars = ax.barh(resp.index, resp.values,
                           color=["#6366f1","#818cf8","#a5b4fc"][:len(resp)])
            ax.set_xlabel("Response Rate (%)", color="#9ca3af", fontsize=8)
            ax.tick_params(colors="#9ca3af", labelsize=8)
            for bar, val in zip(bars, resp.values):
                ax.text(val+.3, bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va='center', color="#e2e8f0", fontsize=8)
            style_ax(ax)
            st.pyplot(fig, use_container_width=True); plt.close()

        with c2:
            if "Recommendation_Category" in camp_df.columns:
                st.markdown("**Recommendation Category Distribution**")
                rec_counts = camp_df["Recommendation_Category"].value_counts()
                fig, ax = dark_fig((5,3))
                palette = ["#6366f1","#34d399","#fbbf24","#f87171","#a78bfa"]
                wedges, texts, autotexts = ax.pie(
                    rec_counts.values, labels=rec_counts.index, autopct="%1.0f%%",
                    colors=palette[:len(rec_counts)],
                    textprops={"color":"#e2e8f0","fontsize":8},
                    wedgeprops={"linewidth":1.5,"edgecolor":"#0f1117"}
                )
                for at in autotexts: at.set_color("#0f1117"); at.set_fontweight("bold")
                st.pyplot(fig, use_container_width=True); plt.close()
            else:
                st.info("No Recommendation_Category column in this dataset.")

        # Segment breakdown table
        st.markdown("**Segment Summary**")
        seg_sum = camp_df.groupby("Segment").agg(
            Customers=("Actual_Label","count"),
            Campaign_Response_Rate=("Campaign_Response","mean"),
        ).round(3)
        st.dataframe(seg_sum.style.format({"Campaign_Response_Rate":"{:.1%}"}),
                     use_container_width=True)
    else:
        st.info("Campaign columns (Segment, Campaign_Response) not found in uploaded data. Showing default dataset.")
        resp = default_predictions.groupby("Segment")["Campaign_Response"].mean().sort_values(ascending=True)*100
        fig, ax = dark_fig((7,3))
        bars = ax.barh(resp.index, resp.values, color=["#6366f1","#818cf8","#a5b4fc"][:len(resp)])
        ax.set_xlabel("Response Rate (%)", color="#9ca3af", fontsize=8)
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for bar, val in zip(bars, resp.values):
            ax.text(val+.3, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va='center', color="#e2e8f0", fontsize=8)
        style_ax(ax)
        st.pyplot(fig, use_container_width=True); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — ASSOCIATION RULES
# ──────────────────────────────────────────────────────────────────────────────
with tab_rules:
    st.markdown('<div class="section-header">Association Rules</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: min_conf = st.slider("Min confidence", 0.0, 1.0, 0.05, 0.01, key="r_conf")
    with c2: min_lift = st.slider("Min lift", 0.0, float(default_rules["lift"].max()), 0.5, 0.05, key="r_lift")
    with c3: top_n   = st.slider("Show top N rules", 5, 25, 10, key="r_topn")

    filtered_rules = default_rules[
        (default_rules["confidence"] >= min_conf) &
        (default_rules["lift"]       >= min_lift)
    ].sort_values("lift", ascending=False).head(top_n)

    disp_cols   = ["antecedents_str","consequents_str","confidence","lift","support"]
    disp_labels = {"antecedents_str":"Antecedents","consequents_str":"Consequents",
                   "confidence":"Confidence","lift":"Lift","support":"Support"}

    st.dataframe(
        filtered_rules[disp_cols].rename(columns=disp_labels)
            .style.format({"Confidence":"{:.3f}","Lift":"{:.3f}","Support":"{:.4f}"})
            .background_gradient(subset=["Lift"], cmap="Blues"),
        use_container_width=True, height=300
    )

    # Lift bar chart
    if not filtered_rules.empty:
        st.markdown("**Top Rules by Lift**")
        top_rules = filtered_rules.head(10)
        labels_bar = [f"{a} → {c}" for a,c in zip(
            top_rules["antecedents_str"], top_rules["consequents_str"])]
        fig, ax = dark_fig((8, max(3, len(labels_bar)*0.45)))
        ax.barh(labels_bar[::-1], top_rules["lift"].values[::-1], color="#6366f1", alpha=0.85)
        ax.set_xlabel("Lift", color="#9ca3af", fontsize=8)
        ax.tick_params(colors="#9ca3af", labelsize=7)
        style_ax(ax)
        st.pyplot(fig, use_container_width=True); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — CUSTOMER EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
with tab_explorer:
    st.markdown('<div class="section-header">Customer Explorer</div>', unsafe_allow_html=True)

    explore_df = default_predictions.copy()
    explore_df["Dynamic_Pred"] = (explore_df["Predicted_Probability"] >= threshold).astype(int)
    explore_df["Correct"] = explore_df["Actual_Label"] == explore_df["Dynamic_Pred"]

    seg_opts = explore_df["Segment"].unique().tolist() if "Segment" in explore_df.columns else []
    if seg_opts:
        seg_filter = st.multiselect("Filter by segment", options=seg_opts, default=seg_opts)
        explore_df = explore_df[explore_df["Segment"].isin(seg_filter)]

    if "Predicted_Probability" in explore_df.columns:
        prob_range = st.slider("Filter by probability range", 0.0, 1.0, (0.0, 1.0), 0.01)
        explore_df = explore_df[
            (explore_df["Predicted_Probability"] >= prob_range[0]) &
            (explore_df["Predicted_Probability"] <= prob_range[1])
        ]

    st.caption(f"Showing {len(explore_df)} customers")

    show_cols = [c for c in ["Customer_ID","Actual_Label","Dynamic_Pred","Predicted_Probability",
                              "Segment","Campaign_Response","Recommendation_Category","Correct"]
                 if c in explore_df.columns]
    rename_map = {"Dynamic_Pred":"Predicted","Predicted_Probability":"Probability"}

    st.dataframe(
        explore_df[show_cols].rename(columns=rename_map)
            .style.format({"Probability":"{:.3f}"})
            .map(style_correct, subset=["Correct"] if "Correct" in show_cols else []),
        use_container_width=True, height=400
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — EXPORT
# ──────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Predictions CSV**")
        exp_df = default_predictions.copy()
        exp_df["Dynamic_Pred"] = (exp_df["Predicted_Probability"] >= threshold).astype(int)
        csv_pred = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download predictions", data=csv_pred,
                           file_name="predictions_filtered.csv", mime="text/csv",
                           use_container_width=True)

    with c2:
        st.markdown("**Filtered Rules CSV**")
        disp_cols2   = ["antecedents_str","consequents_str","confidence","lift","support"]
        disp_labels2 = {"antecedents_str":"Antecedents","consequents_str":"Consequents",
                        "confidence":"Confidence","lift":"Lift","support":"Support"}
        csv_rules = default_rules[disp_cols2].rename(columns=disp_labels2).to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download rules", data=csv_rules,
                           file_name="association_rules_filtered.csv", mime="text/csv",
                           use_container_width=True)

    if "upload_results" in st.session_state:
        st.markdown("---")
        st.markdown("**Uploaded Data — Model Results**")
        results_up = st.session_state["upload_results"]
        test_df_up = st.session_state["upload_test_df"]
        y_true_up  = test_df_up["Actual_Label"]

        export_rows = []
        for mname, res in results_up.items():
            met = compute_metrics(y_true_up, res["y_pred"], res["y_prob"])
            export_rows.append({
                "Model": mname,
                "Accuracy":  met["acc"], "Precision": met["prec"],
                "Recall":    met["rec"], "F1":        met["f1"],
                "AUC":       met["roc_auc"], "Specificity": met["spec"],
            })
        export_df = pd.DataFrame(export_rows)
        csv_up = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download uploaded model results", data=csv_up,
                           file_name="uploaded_model_results.csv", mime="text/csv",
                           use_container_width=True)
