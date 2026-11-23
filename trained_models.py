import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
INPUTS = os.path.join(HERE, "model_inputs")

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

XGB_REGRESSION = {"XGBoost", "Averaging (Voting)", "Stacking (non-neg LR)"}


def to_logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


sys.modules["__main__"].to_logit = to_logit


def available():
    return all(os.path.isdir(os.path.join(MODELS, f"models_{k}")) for k in ["clf", "reg", "ts"]) and os.path.isdir(INPUTS)


def fname(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("/", "_") + ".joblib"


def path(task, name):
    return os.path.join(MODELS, f"models_{task}", fname(name))


@st.cache_data
def specs(task):
    return json.load(open(os.path.join(INPUTS, f"{task}_model_specs.json")))


@st.cache_data
def inventory(task):
    rows = []
    for name, info in specs(task).items():
        p = path(task, name)
        if os.path.exists(p):
            rows.append({"Model": name, "Family": info["family"], "Input": info.get("data") or "raw",
                         "File size (MB)": round(os.path.getsize(p) / 1e6, 2)})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading the trained model...")
def load(task, name):
    return joblib.load(path(task, name))


@st.cache_resource
def preprocessors():
    pre = joblib.load(os.path.join(INPUTS, "preprocessors.joblib"))
    meta = json.load(open(os.path.join(INPUTS, "meta.json")))
    extra = json.load(open(os.path.join(INPUTS, "extra.json")))
    return pre, meta, extra


@st.cache_data
def test_inputs(task):
    X = pd.read_parquet(os.path.join(INPUTS, f"{task}_X_test_raw.parquet"))
    y = pd.read_parquet(os.path.join(INPUTS, f"{task}_y_test.parquet"))["y"].values
    return X, y

def transform(task, kind, raw):
    pre, meta, _ = preprocessors()
    raw = raw[meta["features"]].astype(float)
    if kind in (None, "raw"):
        return raw
    imp = pd.DataFrame(pre[f"imp_{task}"].transform(raw), columns=raw.columns, index=raw.index)
    for c in meta["indicator_cols"]:
        imp[c + "_was_missing"] = raw[c].isna().astype(int)
    if kind == "imputed":
        return imp
    return pd.DataFrame(pre[f"sc_{task}"].transform(imp), columns=imp.columns, index=imp.index)

def predict_proba(name, raw):
    m = load("clf", name)
    return m.predict_proba(transform("clf", specs("clf")[name]["data"], raw))[:, 1]

def predict_change(name, raw, w_rf=0.65):
    if name == "H1 RF+ANN residual":
        m, (_, _, extra) = load("reg", name), preprocessors()
        rf = m["rf"].predict(transform("reg", "imputed", raw))
        ann_in = np.column_stack([transform("reg", "scaled", raw).values, rf / extra["h1_delta_std"]])
        return rf + m["ann_residual"].predict(ann_in)
    if name == "H3 RF+ANN blend":
        return w_rf * predict_change("Random Forest", raw) + (1 - w_rf) * predict_change("ANN (MLP)", raw)
    m = load("reg", name)
    return m.predict(transform("reg", specs("reg")[name]["data"], raw))

def clf_choices():
    return [n for n in specs("clf") if n != "Persistence" and os.path.exists(path("clf", n))]

def reg_choices():
    names = [n for n in specs("reg") if n not in ("Persistence",) and n not in XGB_REGRESSION
             and (n == "H3 RF+ANN blend" or os.path.exists(path("reg", n)))]
    return names
