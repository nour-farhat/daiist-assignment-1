"""
Dashboard for the CHD prevention programme.
Loads the models and files saved by train.ipynb in artifacts/. Nothing is trained here.
Run with: uv run python main.py app
"""

import json
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage pings to Gradio's servers

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score

ART_DIR = "artifacts"
DATA_PATH = "data/framingham.csv"
TARGET = "TenYearCHD"
MODELS = ["scikit-learn", "manual PyTorch", "nn.Module + optim"]
TIER_ORDER = ["low", "moderate", "high"]
ACTIONS = {"low": "standard policy", "moderate": "light plan", "high": "full prevention programme"}

# ---------- load what training saved ----------
with open(f"{ART_DIR}/config.json") as f:
    config = json.load(f)
FEATURES = config["features"]
FILL = config["fill_values"]
DEFAULT_COSTS = config["costs"]

scaler = joblib.load(f"{ART_DIR}/scaler.joblib")
sk_model = joblib.load(f"{ART_DIR}/sklearn_model.joblib")
manual = torch.load(f"{ART_DIR}/manual_torch.pt")
nn_model = nn.Sequential(nn.Linear(len(FEATURES), 1), nn.Sigmoid())
nn_model.load_state_dict(torch.load(f"{ART_DIR}/nn_model.pt"))
nn_model.eval()

test = pd.read_csv(f"{ART_DIR}/test_set.csv")
raw = pd.read_csv(DATA_PATH)
y_test = test[TARGET].values


def predict_all(X):
    # risk from each of the three saved models
    Xs = scaler.transform(X[FEATURES])
    Xt = torch.tensor(Xs, dtype=torch.float32)
    with torch.no_grad():
        return {
            "scikit-learn": sk_model.predict_proba(Xs)[:, 1],
            "manual PyTorch": torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy(),
            "nn.Module + optim": nn_model(Xt).squeeze(-1).numpy(),
        }


P_TEST = predict_all(test)  # recomputed from the loaded models


# ---------- business logic (same formulas as training) ----------
def cutoffs(claim, prog, prog_eff, light, light_eff):
    t_low = light / (light_eff * claim)
    t_high = (prog - light) / ((prog_eff - light_eff) * claim)
    return t_low, t_high


def tiers(p, t_low, t_high):
    return np.where(p >= t_high, "high", np.where(p >= t_low, "moderate", "low"))


def policy_cost(y, p, t_full, t_light, claim, prog, prog_eff, light, light_eff):
    full = p >= t_full
    lite = (p >= t_light) & ~full
    none = ~full & ~lite
    return (np.sum(full * (prog + y * (1 - prog_eff) * claim))
            + np.sum(lite * (light + y * (1 - light_eff) * claim))
            + np.sum(none * y * claim))


def action_cost(p, tier, claim, prog, prog_eff, light, light_eff):
    # expected 10-year cost of the recommended action for someone with risk p
    return np.where(tier == "high", prog + p * (1 - prog_eff) * claim,
                    np.where(tier == "moderate", light + p * (1 - light_eff) * claim, p * claim))


# ---------- tab 1: portfolio ranking ----------
def portfolio(model_name, claim, prog, prog_eff, light, light_eff, tier_filter):
    if prog_eff <= light_eff:
        msg = "**The full programme has to prevent more claims than the light plan.**"
        return msg, None, None, None
    p = P_TEST[model_name]
    t_low, t_high = cutoffs(claim, prog, prog_eff, light, light_eff)
    tier = tiers(p, t_low, t_high)

    df = pd.DataFrame({
        "applicant_id": test["applicant_id"], "risk": p, "tier": tier,
        "action": pd.Series(tier).map(ACTIONS).values,
        "age": test["age"], "male": test["male"], "sysBP": test["sysBP"],
        "cigsPerDay": test["cigsPerDay"], "risk_factors": test["risk_factor_count"],
        "expected cost if no action": p * claim,
        "expected cost with action": action_cost(p, tier, claim, prog, prog_eff, light, light_eff),
        "actual CHD": y_test,
    })
    df["expected saving"] = df["expected cost if no action"] - df["expected cost with action"]
    df = df.sort_values("risk", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    summary = df.groupby("tier").agg(
        applicants=("risk", "size"), mean_predicted_risk=("risk", "mean"),
        actual_CHD_rate=("actual CHD", "mean"), mean_risk_factors=("risk_factors", "mean"),
        total_expected_saving=("expected saving", "sum"),
    ).reindex(TIER_ORDER).reset_index()
    summary["share"] = summary["applicants"] / len(df)
    summary["action"] = summary["tier"].map(ACTIONS)

    n = len(df)
    args = (claim, prog, prog_eff, light, light_eff)
    pol = pd.DataFrame({
        "policy": ["do nothing", "light plan for all", "full programme for all", "model: 3 tiers"],
        "cost per applicant (EUR)": [
            policy_cost(y_test, p, 2, 2, *args) / n,
            policy_cost(y_test, p, 2, 0, *args) / n,
            policy_cost(y_test, p, 0, 0, *args) / n,
            policy_cost(y_test, p, t_high, t_low, *args) / n,
        ],
    })
    fig = px.bar(pol, x="policy", y="cost per applicant (EUR)", text_auto=".0f",
                 title="Real cost on the test set (actual CHD outcomes)")
    fig.update_layout(height=360)

    text = (f"**Cut-offs from these costs:** light plan from **{t_low:.1%}** risk, "
            f"full programme from **{t_high:.1%}**.  \n"
            f"**Saving vs doing nothing:** EUR {pol.iloc[0, 1] - pol.iloc[3, 1]:,.0f} per applicant "
            f"(EUR {(pol.iloc[0, 1] - pol.iloc[3, 1]) * n:,.0f} on these {n} applicants).")

    shown = df if tier_filter == "all" else df[df["tier"] == tier_filter]
    money = ["expected cost if no action", "expected cost with action", "expected saving"]
    shown = shown.drop(columns="actual CHD").round(3)
    shown[money] = shown[money].round(0)
    summary["total_expected_saving"] = summary["total_expected_saving"].round(0)
    return text, summary.round(3), fig, shown


# ---------- tab 2: score a new applicant ----------
def score_applicant(sex, age, education, cigs, bpmeds, stroke, hyp, diabetes, chol, sysbp, glucose,
                    claim, prog, prog_eff, light, light_eff):
    X = pd.DataFrame([{
        "male": int(sex == "male"), "age": age, "education": float(education[0]),
        "cigsPerDay": cigs, "BPMeds": int(bpmeds), "prevalentStroke": int(stroke),
        "prevalentHyp": int(hyp), "diabetes": int(diabetes), "totChol": chol,
        "sysBP": sysbp, "glucose": glucose,
    }])
    for lvl in [2, 3, 4]:
        X[f"edu_{lvl}"] = int(X["education"].iloc[0] == lvl)
    X["log_glucose"] = np.log(X["glucose"])
    X["age_x_sysBP"] = X["age"] * X["sysBP"] / 100
    for c in FEATURES:  # anything the form doesn't ask for gets the test-set median
        if c not in X:
            X[c] = test[c].median()

    preds = predict_all(X)
    p = float(preds["scikit-learn"][0])
    t_low, t_high = cutoffs(claim, prog, prog_eff, light, light_eff)
    tier = tiers(np.array([p]), t_low, t_high)[0]
    no_action = p * claim
    with_action = float(action_cost(np.array([p]), np.array([tier]), claim, prog, prog_eff, light, light_eff)[0])
    rank = int((P_TEST["scikit-learn"] > p).sum()) + 1

    out = pd.DataFrame({"model": MODELS, "10-year CHD risk": [f"{preds[m][0]:.1%}" for m in MODELS]})
    text = (f"## {tier.upper()} risk → {ACTIONS[tier]}\n"
            f"Predicted 10-year CHD risk **{p:.1%}** (average applicant {raw[TARGET].mean():.1%}). "
            f"Would rank **#{rank}** out of {len(test) + 1} applicants.\n\n"
            f"Expected 10-year claim cost: **EUR {no_action:,.0f}** with no action, "
            f"**EUR {with_action:,.0f}** with the recommended action "
            f"(expected saving EUR {no_action - with_action:,.0f}).")
    return out, text


# ---------- tab 3: cut-off, confusion matrix, business cost ----------
def threshold_view(model_name, t_full, t_light, claim, prog, prog_eff, light, light_eff):
    p = P_TEST[model_name]
    t_light = min(t_light, t_full)
    pred = (p >= t_full).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    cm_fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                       x=["not in full programme", "in full programme"], y=["actual no CHD", "actual CHD"],
                       title=f"Full programme at cut-off {t_full:.3f}")
    cm_fig.update_layout(height=380, coloraxis_showscale=False)

    n = len(y_test)
    args = (claim, prog, prog_eff, light, light_eff)
    cost = policy_cost(y_test, p, t_full, t_light, *args)
    nothing = policy_cost(y_test, p, 2, 2, *args)
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    text = (f"**Business cost at these cut-offs:** EUR {cost:,.0f} (EUR {cost / n:,.0f} per applicant)  \n"
            f"**Saving vs doing nothing:** EUR {nothing - cost:,.0f}  \n"
            f"**Full programme:** {pred.mean():.1%} of applicants · catches {recall:.1%} of future CHD cases · "
            f"{precision:.1%} of those enrolled really develop CHD")

    _, t_high = cutoffs(*args)
    ts = np.linspace(0, 0.6, 121)
    curve = [policy_cost(y_test, p, x, min(t_light, x), *args) / n for x in ts]
    cost_fig = go.Figure(go.Scatter(x=ts, y=curve, mode="lines"))
    cost_fig.add_vline(x=t_full, line_color="red", annotation_text="your cut-off")
    cost_fig.add_vline(x=t_high, line_dash="dash", annotation_text=f"theory {t_high:.2f}",
                       annotation_position="bottom right")
    cost_fig.update_layout(title="Cost per applicant vs full-programme cut-off (test set)",
                           xaxis_title="cut-off", yaxis_title="EUR per applicant", height=380)
    return cm_fig, text, cost_fig


# ---------- tab 4: model comparison ----------
def metrics_table():
    base_p = np.full(len(y_test), y_test.mean())
    rows = [{"model": "naive baseline", "ROC-AUC": 0.5,
             "log loss": log_loss(y_test, base_p), "Brier": brier_score_loss(y_test, base_p)}]
    for m in MODELS:
        p = P_TEST[m]
        rows.append({"model": m, "ROC-AUC": roc_auc_score(y_test, p),
                     "log loss": log_loss(y_test, p), "Brier": brier_score_loss(y_test, p)})
    return pd.DataFrame(rows).round(4)


def calibration_plot():
    fig = go.Figure(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines", name="perfect",
                               line=dict(dash="dash", color="grey")))
    for m in MODELS:
        d = pd.DataFrame({"p": P_TEST[m], "y": y_test})
        d["bin"] = pd.qcut(d["p"], 10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["actual"], mode="lines+markers", name=m))
    fig.update_layout(title="Predicted risk vs actual CHD rate (test set, 10 groups)",
                      xaxis_title="mean predicted risk", yaxis_title="actual CHD rate", height=420)
    return fig


def agreement_plot():
    ref = P_TEST["scikit-learn"]
    fig = go.Figure()
    for m in MODELS[1:]:
        fig.add_trace(go.Scatter(x=ref, y=P_TEST[m], mode="markers", name=m, opacity=0.5))
    fig.add_trace(go.Scatter(x=[0, ref.max()], y=[0, ref.max()], mode="lines", name="identical",
                             line=dict(dash="dash", color="grey")))
    fig.update_layout(title="Do the three implementations agree? (one dot = one applicant)",
                      xaxis_title="scikit-learn risk", yaxis_title="PyTorch risk", height=420)
    return fig


def risk_by_outcome(model_name):
    d = pd.DataFrame({"predicted risk": P_TEST[model_name],
                      "actual outcome": np.where(y_test == 1, "CHD", "no CHD")})
    fig = px.box(d, x="actual outcome", y="predicted risk", points="all",
                 title=f"Predicted risk by actual outcome ({model_name})")
    fig.update_layout(height=420)
    return fig


def coef_table():
    return pd.read_csv(f"{ART_DIR}/coefficients.csv").rename(columns={"Unnamed: 0": "feature"}).round(4)


# ---------- tab 5: distributions ----------
DIST_COLS = ["age", "sysBP", "diaBP", "totChol", "BMI", "heartRate", "glucose", "cigsPerDay",
             "male", "education", "currentSmoker", "BPMeds", "prevalentHyp", "diabetes", "prevalentStroke"]


def feature_dist(col, log_x):
    d = raw[[col, TARGET]].dropna().copy()
    d["outcome"] = np.where(d[TARGET] == 1, "CHD", "no CHD")
    if d[col].nunique() <= 5:
        rate = d.groupby(col)[TARGET].mean().reset_index()
        fig = px.bar(rate, x=col, y=TARGET, title=f"CHD rate by {col}")
        fig.add_hline(y=raw[TARGET].mean(), line_dash="dash", annotation_text="overall rate")
        fig.update_layout(yaxis_title="CHD rate")
    else:
        if log_x:
            d[col] = np.log(d[col].clip(lower=0.5))
        fig = px.histogram(d, x=col, color="outcome", barmode="overlay", histnorm="probability density",
                           nbins=50, opacity=0.6, title=f"{'log ' if log_x else ''}{col} by 10-year CHD outcome")
    fig.update_layout(height=420)
    return fig


def target_dist():
    counts = raw[TARGET].map({0: "no CHD", 1: "CHD"}).value_counts().reset_index()
    counts.columns = ["outcome", "people"]
    fig = px.bar(counts, x="outcome", y="people", text="people",
                 title=f"Target: 10-year CHD ({raw[TARGET].mean():.1%} positive)")
    fig.update_layout(height=420)
    return fig


# ---------- layout ----------
with gr.Blocks(title="CHD prevention programme") as demo:
    gr.Markdown(
        "# Heart-risk prevention programme: applicant triage\n"
        "An insurer ranks applicants by predicted 10-year coronary heart disease (CHD) risk and offers "
        "**high** risk the full prevention programme, **moderate** risk a light plan, **low** risk the standard policy. "
        "Logistic regression trained three ways; everything is loaded from `artifacts/`, nothing is retrained."
    )

    with gr.Accordion("Cost assumptions (change them to see the tiers move)", open=False):
        with gr.Row():
            c_claim = gr.Number(DEFAULT_COSTS["claim"], label="cost of one CHD claim (EUR)")
            c_prog = gr.Number(DEFAULT_COSTS["programme"], label="full programme per person (EUR)")
            c_prog_eff = gr.Slider(0.05, 0.8, DEFAULT_COSTS["programme_effect"], step=0.01,
                                   label="claims prevented by full programme")
            c_light = gr.Number(DEFAULT_COSTS["light"], label="light plan per person (EUR)")
            c_light_eff = gr.Slider(0.0, 0.5, DEFAULT_COSTS["light_effect"], step=0.01,
                                    label="claims prevented by light plan")
    cost_inputs = [c_claim, c_prog, c_prog_eff, c_light, c_light_eff]

    with gr.Tab("1. Applicant ranking"):
        with gr.Row():
            r_model = gr.Dropdown(MODELS, value="scikit-learn", label="model")
            r_filter = gr.Dropdown(["all"] + TIER_ORDER, value="all", label="show tier")
        r_text = gr.Markdown()
        r_summary = gr.Dataframe(label="Tiers on the test applicants")
        r_fig = gr.Plot()
        r_table = gr.Dataframe(label="Applicants ranked by predicted risk")
        r_inputs = [r_model] + cost_inputs + [r_filter]
        r_outputs = [r_text, r_summary, r_fig, r_table]
        for comp in r_inputs:
            comp.change(portfolio, r_inputs, r_outputs)
        demo.load(portfolio, r_inputs, r_outputs)

    with gr.Tab("2. Score a new applicant"):
        with gr.Row():
            with gr.Column():
                a_sex = gr.Radio(["female", "male"], value="male", label="sex")
                a_age = gr.Slider(32, 70, value=55, step=1, label="age")
                a_edu = gr.Dropdown(["1: some high school", "2: high school / GED",
                                     "3: some college", "4: college graduate"],
                                    value="2: high school / GED", label="education")
                a_cigs = gr.Slider(0, 60, value=10, step=1, label="cigarettes per day")
                a_chol = gr.Slider(100, 450, value=240, step=1, label="total cholesterol (mg/dL)")
                a_sys = gr.Slider(85, 230, value=140, step=1, label="systolic blood pressure (mmHg)")
                a_glu = gr.Slider(40, 400, value=85, step=1, label="glucose (mg/dL)")
            with gr.Column():
                a_bp = gr.Checkbox(label="on blood pressure medication")
                a_stroke = gr.Checkbox(label="previous stroke")
                a_hyp = gr.Checkbox(label="hypertensive")
                a_diab = gr.Checkbox(label="diabetes")
                a_btn = gr.Button("Score applicant", variant="primary")
                a_text = gr.Markdown()
                a_table = gr.Dataframe(label="Risk from each saved model")
        a_inputs = [a_sex, a_age, a_edu, a_cigs, a_bp, a_stroke, a_hyp, a_diab, a_chol, a_sys, a_glu] + cost_inputs
        a_btn.click(score_applicant, a_inputs, [a_table, a_text])

    with gr.Tab("3. Cut-off, confusion matrix & cost"):
        t_low0, t_high0 = cutoffs(*[DEFAULT_COSTS[k] for k in
                                    ["claim", "programme", "programme_effect", "light", "light_effect"]])
        with gr.Row():
            t_model = gr.Dropdown(MODELS, value="scikit-learn", label="model")
            t_full = gr.Slider(0.0, 0.6, value=round(t_high0, 3), step=0.005,
                               label="full programme if risk ≥")
            t_light = gr.Slider(0.0, 0.6, value=round(t_low0, 3), step=0.005,
                                label="light plan if risk ≥")
        with gr.Row():
            cm_plot = gr.Plot()
            cost_plot = gr.Plot()
        cost_text = gr.Markdown()
        t_inputs = [t_model, t_full, t_light] + cost_inputs
        t_outputs = [cm_plot, cost_text, cost_plot]
        for comp in t_inputs:
            comp.change(threshold_view, t_inputs, t_outputs)
        demo.load(threshold_view, t_inputs, t_outputs)

    with gr.Tab("4. Compare the three models"):
        gr.Dataframe(metrics_table(), label="Test set metrics vs naive baseline")
        with gr.Row():
            gr.Plot(calibration_plot())
            gr.Plot(agreement_plot())
        m_box = gr.Dropdown(MODELS, value="scikit-learn", label="model")
        m_plot = gr.Plot(risk_by_outcome("scikit-learn"))
        m_box.change(risk_by_outcome, m_box, m_plot)
        gr.Dataframe(coef_table(), label="Coefficients (scaled features) from each implementation")

    with gr.Tab("5. Feature & target distributions"):
        with gr.Row():
            d_col = gr.Dropdown(DIST_COLS, value="glucose", label="feature")
            d_log = gr.Checkbox(label="log scale", value=False)
        d_plot = gr.Plot(feature_dist("glucose", False))
        d_col.change(feature_dist, [d_col, d_log], d_plot)
        d_log.change(feature_dist, [d_col, d_log], d_plot)
        gr.Plot(target_dist())


if __name__ == "__main__":
    demo.launch()
