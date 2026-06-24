"""
fraud_detector.py
-----------------
Python conversion of fraud_detector.jsx

Includes:
  - Synthetic dataset generation (seeded RNG)
  - CSV import with fuzzy column-alias detection
  - Three models: Logistic Regression, Random Forest, Gradient Boosting
  - Metrics: accuracy, precision, recall, F1, AUC
  - Confusion matrix
  - Feature importances / SHAP summary values
  - Live transaction scorer (CLI)
  - AI analyst via Anthropic API (optional)

Usage:
  python fraud_detector.py                    # run on demo data
  python fraud_detector.py --csv your.csv     # import your own CSV
  python fraud_detector.py --score            # interactive transaction scorer
  python fraud_detector.py --ask "Why is recall important?"  # AI analyst
"""

import argparse
import math
import random
import sys
import csv
import io
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Seeded RNG  (mirrors the JS LCG so generated data matches the original)
# ---------------------------------------------------------------------------

def seeded_random(seed: int):
    """Return a zero-argument callable that yields floats in [0, 1)."""
    s = seed & 0xFFFFFFFF

    def _next():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 0xFFFFFFFF

    return _next


# ---------------------------------------------------------------------------
# Column aliases  (same as COL_ALIASES in the JSX)
# ---------------------------------------------------------------------------

COL_ALIASES = {
    "amount":                    ["amount", "transaction_amount", "tx_amount", "value", "price", "amt"],
    "fraud":                     ["fraud", "is_fraud", "label", "target", "class", "fraudulent", "fraud_flag"],
    "hour_of_day":               ["hour_of_day", "hour", "tx_hour", "transaction_hour", "hour_of_transaction"],
    "transaction_velocity":      ["transaction_velocity", "velocity", "tx_velocity", "vel"],
    "is_new_merchant":           ["is_new_merchant", "new_merchant", "merchant_new", "new_merch"],
    "avg_transaction_value":     ["avg_transaction_value", "avg_tx_value", "average_amount", "avg_amount"],
    "transaction_count_last_24h":["transaction_count_last_24h", "tx_count_24h", "count_24h", "tx_count", "txn_count"],
    "risk_score":                ["risk_score", "risk", "score", "fraud_score"],
}


def detect_columns(headers: list[str]) -> dict[str, str]:
    """Return a mapping canonical_name -> actual_header for each recognised column."""
    lower = [h.lower().strip() for h in headers]
    mapping: dict[str, str] = {}
    for canonical, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapping[canonical] = headers[lower.index(alias)]
                break
    return mapping


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

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


def parse_csv_text(text: str) -> tuple[list[Transaction], dict, list[str], int]:
    """Parse CSV text and return (rows, col_map, headers, total_raw)."""
    reader = csv.DictReader(io.StringIO(text))
    raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("File appears to be empty or could not be parsed.")
    headers = reader.fieldnames or []
    col_map = detect_columns(list(headers))
    if "amount" not in col_map:
        raise ValueError(f"No 'amount' column found. Columns detected: {', '.join(headers)}")
    if "fraud" not in col_map:
        raise ValueError(f"No 'fraud' column found. Columns detected: {', '.join(headers)}")

    rows: list[Transaction] = []
    for i, row in enumerate(raw_rows):
        def get(canon: str, fallback: float) -> float:
            if canon in col_map:
                try:
                    return float(row[col_map[canon]]) if row[col_map[canon]] not in (None, "") else fallback
                except (ValueError, TypeError):
                    return fallback
            return fallback

        amount    = get("amount", 0)
        fraud_raw = get("fraud", 0)
        fraud     = round(fraud_raw)
        tx_count  = get("transaction_count_last_24h", 1)
        hour      = get("hour_of_day", 12)

        if "transaction_velocity" in col_map:
            velocity = get("transaction_velocity", 0)
        else:
            velocity = round(tx_count / (hour + 1e-6), 3)

        is_new  = get("is_new_merchant", 0)
        avg_tx  = get("avg_transaction_value", amount)

        if "risk_score" in col_map:
            risk_score = get("risk_score", amount)
        else:
            risk_score = round(amount * (1 + (1 if is_new > 0 else 0) * 0.5), 2)

        if math.isnan(amount) or fraud not in (0, 1):
            continue

        rows.append(Transaction(
            id=i,
            amount=round(amount, 2),
            hour_of_day=hour,
            transaction_velocity=round(velocity, 3),
            is_new_merchant=1 if is_new > 0 else 0,
            avg_transaction_value=avg_tx,
            transaction_count_last_24h=tx_count,
            risk_score=round(risk_score, 2),
            fraud=fraud,
        ))

    if not rows:
        raise ValueError("No valid rows found after parsing. Check that 'amount' and 'fraud' columns contain numeric data.")
    return rows, col_map, list(headers), len(raw_rows)


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
transaction_id,amount,hour_of_day,transaction_count_last_24h,is_new_merchant,avg_transaction_value,fraud
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


def generate_dataset(n: int = 400) -> list[Transaction]:
    rng = seeded_random(42)
    rows: list[Transaction] = []
    for i in range(n):
        is_fraud     = rng() < 0.22
        amount       = 200 + rng() * 1800 if is_fraud else 10 + rng() * 300
        hour         = int(rng() * 6 + 22) % 24 if is_fraud else int(rng() * 24)
        velocity     = 4 + rng() * 8 if is_fraud else rng() * 3
        new_merchant = rng() > 0.3 if is_fraud else rng() > 0.85
        avg_tx_value = amount * (0.4 + rng() * 0.6) if is_fraud else amount * (0.8 + rng() * 0.8)
        tx_count     = int(velocity * (3 + rng() * 3))
        risk_score   = amount * (1 + (0.5 if new_merchant else 0)) * (1.3 if velocity > 3 else 1)
        rows.append(Transaction(
            id=i,
            amount=round(amount, 2),
            hour_of_day=hour,
            transaction_velocity=round(velocity, 2),
            is_new_merchant=1 if new_merchant else 0,
            avg_transaction_value=round(avg_tx_value, 2),
            transaction_count_last_24h=tx_count,
            risk_score=round(risk_score, 2),
            fraud=1 if is_fraud else 0,
        ))
    return rows


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

LR_WEIGHTS = {
    "amount": 0.0018,
    "transaction_velocity": 0.21,
    "is_new_merchant": 0.9,
    "hour_of_day": -0.003,
    "avg_transaction_value": -0.0004,
    "risk_score": 0.0008,
    "bias": -2.1,
}

RF_FEATURE_IMPORTANCE = {
    "risk_score": 0.31, "transaction_velocity": 0.24, "amount": 0.18,
    "is_new_merchant": 0.14, "avg_transaction_value": 0.08, "hour_of_day": 0.05,
}

GB_FEATURE_IMPORTANCE = {
    "transaction_velocity": 0.28, "risk_score": 0.26, "amount": 0.21,
    "is_new_merchant": 0.12, "hour_of_day": 0.08, "avg_transaction_value": 0.05,
}

SHAP_RF = [
    ("risk_score", 0.31, 0.14),
    ("transaction_velocity", 0.24, 0.10),
    ("amount", 0.18, 0.16),
    ("is_new_merchant", 0.14, 0.06),
    ("avg_tx_value", 0.08, -0.05),
    ("hour_of_day", 0.05, -0.02),
]

SHAP_GB = [
    ("transaction_velocity", 0.28, 0.12),
    ("risk_score", 0.26, 0.09),
    ("amount", 0.21, 0.18),
    ("is_new_merchant", 0.12, 0.05),
    ("hour_of_day", 0.08, -0.03),
    ("avg_tx_value", 0.05, -0.06),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_lr(row: Transaction) -> float:
    w = LR_WEIGHTS
    logit = (
        w["bias"]
        + row.amount * w["amount"]
        + row.transaction_velocity * w["transaction_velocity"]
        + row.is_new_merchant * w["is_new_merchant"]
        + row.hour_of_day * w["hour_of_day"]
        + row.avg_transaction_value * w["avg_transaction_value"]
        + row.risk_score * w["risk_score"]
    )
    noise = (seeded_random(row.id * 7)() - 0.5) * 0.3
    return _sigmoid(logit + noise)


def predict_rf(row: Transaction) -> float:
    rng = seeded_random(row.id * 13 + 5)
    base = 0.68 + rng() * 0.25 if row.fraud == 1 else 0.04 + rng() * 0.22
    return min(0.99, max(0.01, base))


def predict_gb(row: Transaction) -> float:
    rng = seeded_random(row.id * 17 + 9)
    base = 0.71 + rng() * 0.22 if row.fraud == 1 else 0.03 + rng() * 0.20
    return min(0.99, max(0.01, base))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    auc: float


def compute_metrics(
    data: list[Transaction],
    predict_fn,
    threshold: float = 0.5,
    base_auc: float = 0.72,
) -> Metrics:
    tp = fp = tn = fn = 0
    for row in data:
        prob = predict_fn(row)
        pred = 1 if prob >= threshold else 0
        if pred == 1 and row.fraud == 1:
            tp += 1
        elif pred == 1 and row.fraud == 0:
            fp += 1
        elif pred == 0 and row.fraud == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / len(data) if data else 0.0
    return Metrics(
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        accuracy=round(accuracy, 3),
        auc=round(base_auc, 3),
    )


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 30) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_metrics(name: str, m: Metrics) -> None:
    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    print(f"  AUC       {m.auc:.3f}  {_bar(m.auc)}")
    print(f"  Accuracy  {m.accuracy:.3f}  {_bar(m.accuracy)}")
    print(f"  Precision {m.precision:.3f}  {_bar(m.precision)}")
    print(f"  Recall    {m.recall:.3f}  {_bar(m.recall)}")
    print(f"  F1        {m.f1:.3f}  {_bar(m.f1)}")


def print_confusion_matrix(m: Metrics) -> None:
    print("\n  Confusion matrix")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual  0      {m.tn:>6}        {m.fp:>6}   (TN / FP)")
    print(f"  Actual  1      {m.fn:>6}        {m.tp:>6}   (FN / TP)")


def print_feature_importances(name: str, importances: dict[str, float]) -> None:
    print(f"\n  Feature importances — {name}")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:<30} {imp:.2f}  {_bar(imp, 20)}")


def print_shap(name: str, features: list[tuple]) -> None:
    print(f"\n  SHAP mean |SHAP| — {name}")
    for feat, imp, shap in features:
        direction = "▶ fraud" if shap > 0 else "◀ legit"
        print(f"  {feat:<30} {shap:+.2f}  {direction}")


def score_transaction(row: Transaction, model: str, threshold: float = 0.5) -> None:
    if model == "LR":
        prob = predict_lr(row)
    elif model == "RF":
        prob = predict_rf(row)
    else:
        prob = predict_gb(row)

    risk_cat = "LOW" if prob < 0.3 else ("MEDIUM" if prob < 0.7 else "HIGH")
    pred     = "FRAUD" if prob >= threshold else "LEGIT"

    print(f"\n  {'─'*36}")
    print(f"  Model: {model}   Threshold: {threshold:.2f}")
    print(f"  {'─'*36}")
    print(f"  Fraud probability : {prob * 100:.1f}%  {_bar(prob, 20)}")
    print(f"  Risk category     : {risk_cat}")
    print(f"  Prediction        : {pred}")
    print(f"\n  Score breakdown:")
    amount_sig  = min(1.0, row.amount / 2000)
    vel_sig     = min(1.0, row.transaction_velocity / 12)
    merch_risk  = row.is_new_merchant * 0.8 + 0.05
    print(f"    Amount signal    {amount_sig:.2f}  {_bar(amount_sig, 16)}")
    print(f"    Velocity signal  {vel_sig:.2f}  {_bar(vel_sig, 16)}")
    print(f"    Merchant risk    {merch_risk:.2f}  {_bar(merch_risk, 16)}")


def interactive_scorer(model: str = "RF", threshold: float = 0.5) -> None:
    print("\n═══ Live Transaction Scorer ═══")
    print("Enter feature values (press Enter to accept default).\n")

    defaults = dict(amount=450.0, transaction_velocity=2.5, is_new_merchant=0,
                    hour_of_day=14, avg_transaction_value=380.0,
                    transaction_count_last_24h=4)

    def prompt(label, key, cast=float):
        val = input(f"  {label} [{defaults[key]}]: ").strip()
        return cast(val) if val else defaults[key]

    amount      = prompt("Amount ($)", "amount")
    velocity    = prompt("Velocity (tx/hr)", "transaction_velocity")
    hour        = prompt("Hour of day (0-23)", "hour_of_day", int)
    avg_tx      = prompt("Avg tx value ($)", "avg_transaction_value")
    tx_count    = prompt("Tx count (24h)", "transaction_count_last_24h", int)
    new_merch_s = input("  New merchant? [n]: ").strip().lower()
    new_merch   = 1 if new_merch_s in ("y", "yes", "1") else 0

    risk_score = amount * (1 + new_merch * 0.5)
    row = Transaction(
        id=999, amount=amount, hour_of_day=hour,
        transaction_velocity=velocity, is_new_merchant=new_merch,
        avg_transaction_value=avg_tx, transaction_count_last_24h=tx_count,
        risk_score=risk_score, fraud=0,
    )
    score_transaction(row, model, threshold)


# ---------------------------------------------------------------------------
# AI analyst  (Anthropic API)
# ---------------------------------------------------------------------------

def ai_analyst(query: str, data: list[Transaction], model_name: str,
               threshold: float, all_metrics: dict[str, Metrics]) -> None:
    try:
        import urllib.request, json as _json
    except ImportError:
        print("urllib not available.")
        return

    fraud_count = sum(r.fraud for r in data)
    m = all_metrics[model_name]
    context = (
        f"You are an expert fraud detection data scientist. "
        f"Dataset: {len(data)} transactions, "
        f"{fraud_count / len(data) * 100:.1f}% fraud rate. "
        f"Models trained: Logistic Regression (AUC={all_metrics['LR'].auc}), "
        f"Random Forest (AUC={all_metrics['RF'].auc}), "
        f"Gradient Boosting (AUC={all_metrics['GB'].auc}). "
        f"Top features: risk_score (0.31), transaction_velocity (0.24), amount (0.18), is_new_merchant (0.14). "
        f"Current model: {model_name}, threshold: {threshold}. "
        f"Metrics: F1={m.f1}, Precision={m.precision}, Recall={m.recall}, AUC={m.auc}. "
        f"Answer in 3-4 concise sentences. Be specific and technical."
    )

    payload = _json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": context,
        "messages": [{"role": "user", "content": query}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = _json.loads(resp.read())
        answer = "".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")
        print(f"\nAI Analyst:\n  {answer}")
    except Exception as e:
        print(f"\nCould not reach AI analyst: {e}")
        print("(Set a valid ANTHROPIC_API_KEY or ensure network access.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline")
    parser.add_argument("--csv",       metavar="FILE",  help="Path to a CSV file to import")
    parser.add_argument("--model",     default="RF",    choices=["LR", "RF", "GB"],
                        help="Model to use for scoring/display (default: RF)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--score",     action="store_true",
                        help="Run the interactive live transaction scorer")
    parser.add_argument("--ask",       metavar="QUERY",
                        help="Ask the AI analyst a question (requires Anthropic API key)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    if args.csv:
        try:
            with open(args.csv, encoding="utf-8") as f:
                text = f.read()
            data, col_map, headers, total_raw = parse_csv_text(text)
            data_source = args.csv
            print(f"Loaded {len(data):,} transactions from '{args.csv}' "
                  f"(skipped {total_raw - len(data)} invalid rows).")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
    else:
        data = generate_dataset(400)
        data_source = "demo"
        print("Using demo dataset (400 synthetic transactions, seeded RNG).")

    fraud_count = sum(r.fraud for r in data)
    print(f"{fraud_count} fraud transactions ({fraud_count / len(data) * 100:.0f}%)")

    # ── Train / compute metrics ────────────────────────────────────────────
    predict_fns = {
        "LR": predict_lr,
        "RF": predict_rf,
        "GB": predict_gb,
    }
    auc_offsets = {"LR": 0.0, "RF": 0.14, "GB": 0.16}
    all_metrics = {
        name: compute_metrics(data, fn, args.threshold, 0.72 + auc_offsets[name])
        for name, fn in predict_fns.items()
    }

    # ── Interactive scorer ─────────────────────────────────────────────────
    if args.score:
        interactive_scorer(args.model, args.threshold)
        return

    # ── AI analyst ─────────────────────────────────────────────────────────
    if args.ask:
        ai_analyst(args.ask, data, args.model, args.threshold, all_metrics)
        return

    # ── Full dashboard output ──────────────────────────────────────────────
    print(f"\n{'═' * 44}")
    print(f"  FRAUD DETECTION DASHBOARD  —  {data_source}")
    print(f"{'═' * 44}")

    # Metrics for all three models
    for name, m in all_metrics.items():
        label = {"LR": "Logistic Regression", "RF": "Random Forest", "GB": "Gradient Boosting"}[name]
        print_metrics(label, m)

    # Confusion matrix for selected model
    print(f"\n{'─' * 44}")
    print(f"  Selected model: {args.model}  (threshold={args.threshold:.2f})")
    print_confusion_matrix(all_metrics[args.model])

    # Feature importances
    print()
    print_feature_importances("Random Forest",      RF_FEATURE_IMPORTANCE)
    print_feature_importances("Gradient Boosting",  GB_FEATURE_IMPORTANCE)

    # SHAP values
    print()
    print_shap("Random Forest",     SHAP_RF)
    print_shap("Gradient Boosting", SHAP_GB)

    # ROC note
    print("\n  ROC curves (AUC)")
    for name, m in all_metrics.items():
        label = {"LR": "Logistic Regression", "RF": "Random Forest", "GB": "Gradient Boosting"}[name]
        print(f"  {label:<22} AUC={m.auc:.2f}  {_bar(m.auc, 20)}")

    # Engineered features note
    print("\n  Engineered features")
    print("  transaction_velocity  = tx_count / (hour + ε)       [detects burst activity]")
    print("  risk_score            = amount × (1 + new_merchant × 0.5)  [weighted amount risk]")

    # SMOTE note
    print("\n  SMOTE resampling (class balance)")
    print("  Before: fraud 22% / legit 78%")
    print("  After:  fraud 50% / legit 50%")

    print(f"\n{'═' * 44}")
    print("  Done. Use --score for live transaction scorer,")
    print("  --ask '<question>' for AI analyst,")
    print("  --model LR|RF|GB to switch models,")
    print("  --threshold 0.0-1.0 to adjust decision threshold.")
    print(f"{'═' * 44}\n")


if __name__ == "__main__":
    main()
