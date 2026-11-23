import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import trained_models as tm

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FAM_COLORS = {"Baseline": "#9AA0A6", "ANN": "#8E7CC3", "Bagging": "#3D7EBF", "Boosting": "#E08A3C",
              "Stacking": "#4E9F6E", "Hybrid": "#C9504A"}
TS_COLORS = {"Persistence": "#9AA0A6", "ARIMA": "#8E7CC3", "RF (direct)": "#3D7EBF", "RF + ARIMA hybrid": "#C9504A"}
STATUS = {0: "Normal", 1: "Alert", 2: "Minor flood", 3: "Major flood"}
STATUS_COLORS = {"Normal": "#4E9F6E", "Alert": "#E3B23C", "Minor flood": "#E08A3C", "Major flood": "#C9504A"}

st.set_page_config(page_title="River Flood Risk", page_icon=":material/water:", layout="wide")

st.markdown("""
<style>
:root { --accent: #3D7EBF; }
[data-testid="stBaseButton-pillsActive"], [data-testid="stBaseButton-segmented_controlActive"] {
    color: var(--accent) !important; border-color: var(--accent) !important;
    background-color: rgba(61, 126, 191, 0.12) !important; }
[data-testid="stBaseButton-pills"]:hover, [data-testid="stBaseButton-segmented_control"]:hover {
    color: var(--accent) !important; border-color: var(--accent) !important; }
span[data-baseweb="tag"] { background-color: var(--accent) !important; }
div[data-baseweb="slider"] div[role="slider"] { background-color: var(--accent) !important; }
[data-testid="stSliderThumbValue"] { color: var(--accent) !important; }
div[data-baseweb="slider"] > div > div { position: relative; }
div[data-baseweb="slider"] > div > div::after {
    content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
    background: var(--accent); mix-blend-mode: hue; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    rd = lambda f: pd.read_csv(os.path.join(DATA, f))
    rp = lambda f: pd.read_parquet(os.path.join(DATA, f))
    d = {
        "clf_val": rd("clf_results_val.csv"), "clf_test": rd("clf_results_test.csv"),
        "clf_ci": rd("clf_results_test_ci.csv"), "clf_onset": rd("clf_results_new_alert.csv"),
        "clf_pred": rp("clf_test_predictions.parquet"),
        "reg_val": rd("reg_results_val.csv"), "reg_test": rd("reg_results_test.csv"),
        "reg_ci": rd("reg_results_test_ci.csv"), "reg_band": rd("reg_results_by_band.csv"),
        "reg_pred": rp("reg_test_predictions.parquet"),
        "ts_val": rd("ts_results_val.csv"), "ts_test": rd("ts_results_test.csv"),
        "ts_ci": rd("ts_results_test_ci.csv"), "ts_station": rd("ts_results_per_station.csv"),
        "ts_orders": rd("ts_arima_orders.csv"), "ts_pred": rp("ts_test_predictions.parquet"),
        "clean": rp("clean_full.parquet"), "imp": rd("feature_importance.csv"),
        "X_test": rp("clf_X_test_raw.parquet"), "ids_test": rp("clf_ids_test.parquet"),
        "clf_reg": json.load(open(os.path.join(DATA, "clf_model_registry.json"))),
        "meta": json.load(open(os.path.join(DATA, "meta.json"))),
    }
    d["clean"]["obs_datetime"] = pd.to_datetime(d["clean"]["obs_datetime"])
    for k in ["clf_pred", "reg_pred", "ts_pred"]:
        d[k]["obs_datetime"] = pd.to_datetime(d[k]["obs_datetime"])
    d["ts_pred"]["target_time"] = pd.to_datetime(d["ts_pred"]["target_time"])
    d["reg_band"] = d["reg_band"].rename(columns={"Unnamed: 0": "Water level band"})
    return d


@st.cache_resource
def load_models():
    from catboost import CatBoostClassifier, CatBoostRegressor
    clf, reg = CatBoostClassifier(), CatBoostRegressor()
    clf.load_model(os.path.join(DATA, "catboost_clf.cbm"))
    reg.load_model(os.path.join(DATA, "catboost_reg.cbm"))
    return clf, reg


if not os.path.exists(os.path.join(DATA, "clf_results_test.csv")):
    st.error(f"No data found in `{DATA}`. Unzip dashboard_bundle.zip (exported from Colab) into "
             "`dashboard/data/` and reload the page.")
    st.stop()
D = load()
CUT_VAL, CUT_TEST = pd.Timestamp(D["meta"]["cut_val"]), pd.Timestamp(D["meta"]["cut_test"])
END = D["clean"].obs_datetime.max()

def bar_chart(df, metric, title, ascending=True, fmt="{:.3f}"):
    t = df.sort_values(metric, ascending=ascending)
    fig = go.Figure(go.Bar(x=t[metric], y=t["Model"], orientation="h", marker_color=t["Family"].map(FAM_COLORS),
                           text=[fmt.format(v) for v in t[metric]], textposition="outside",
                           customdata=t["Family"], hovertemplate="%{y} (%{customdata}): %{x:.4f}<extra></extra>"))
    lo = min(0, t[metric].min() * 1.15)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=60 + 30 * len(t),
                      margin=dict(l=10, r=50, t=45, b=10), xaxis=dict(range=[lo, t[metric].max() * 1.18]))
    return fig


def family_key(families):
    st.markdown(" ".join(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
                         f"background:{FAM_COLORS[f]};margin:0 4px 0 12px'></span>{f}" for f in families),
                unsafe_allow_html=True)


def page_header(title, intro):
    st.title(title)
    st.caption(intro)


def station_picker(label, stations, default="Hanwella", key=None):
    stations = sorted(stations)
    return st.selectbox(label, stations, index=stations.index(default) if default in stations else 0, key=key)


def fmt_table(df, cols4=()):
    return df.style.format({c: "{:.4f}" for c in cols4 if c in df})


def split_choice(key):
    choice = st.segmented_control("Results for", ["Test period", "Validation period"], default="Test period", key=key)
    return "test" if choice != "Validation period" else "val"

def gap(px=24):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def overview():
    c = D["clean"]
    st.title("Flood Risk Prediction Dashboard")
    st.write(f"River gauging stations in Sri Lanka, {c.obs_datetime.min():%B %Y} to {END:%B %Y}. "
             "Source: Irrigation Department water level and rainfall reports.")
    gap(8)

    k = st.columns(4)
    k[0].metric("Reports processed", "1,135")
    k[1].metric("Gauging stations", c.station.nunique())
    k[2].metric("River basins", c.basin.nunique())
    k[3].metric("Readings", f"{len(c):,}")
    gap()

    st.subheader("Test set results")
    best = D["clf_test"].sort_values("PR_AUC", ascending=False).iloc[0]
    pers = D["clf_test"].set_index("Model").loc["Persistence"]
    regb = D["reg_test"].sort_values("RMSE").iloc[0]
    tsb = D["ts_test"].set_index("Model").loc["RF + ARIMA hybrid"]
    onset = D["clf_onset"].sort_values("Recall", ascending=False).iloc[0]
    r = st.columns(4)
    r[0].metric("Best classifier (PR-AUC)", f"{best.PR_AUC:.3f}", f"{best.PR_AUC - pers.PR_AUC:+.3f} vs persistence")
    r[1].metric("New alert recall", f"{onset.Recall:.1%}", "persistence: 0%", delta_color="off")
    r[2].metric("Best level forecast (RMSE)", f"{regb.RMSE:.3f} m", f"{regb.Skill:.1%} skill")
    r[3].metric("RF + ARIMA (RMSE)", f"{tsb.RMSE:.3f} m", f"{tsb.Skill:.1%} skill")
    st.caption(f"Best models: {best.Model} (classification), {onset.Model} (new alerts), {regb.Model} (level forecast). "
               "Skill = reduction in RMSE compared with persistence.")
    gap()

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("Readings per month")
        m = c.assign(month=c.obs_datetime.dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
            readings=("station", "size"), flood=("status", lambda s: int((s >= 2).sum())))
        fig = go.Figure()
        fig.add_bar(x=m.index, y=m.readings, name="All readings", marker_color="#3D7EBF")
        fig.add_bar(x=m.index, y=m.flood, name="Minor/major flood", marker_color="#C9504A")
        fig.add_vrect(x0=CUT_VAL, x1=CUT_TEST, fillcolor="#888", opacity=0.12, line_width=0)
        fig.add_vrect(x0=CUT_TEST, x1=END + pd.Timedelta(days=15), fillcolor="#888", opacity=0.25, line_width=0)
        fig.update_layout(barmode="overlay", height=300, margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig)
        st.caption("Shaded: validation (light) and test (dark) periods.")
    with right:
        st.subheader("Status counts")
        sc = c.status.map(STATUS).value_counts().reindex(list(STATUS.values())).fillna(0).astype(int)
        st.dataframe(pd.DataFrame({"Status": sc.index, "Readings": sc.values,
                                   "Share": (sc.values / sc.sum() * 100).round(1)}),
                     hide_index=True, width="stretch",
                     column_config={"Share": st.column_config.NumberColumn("Share (%)", format="%.1f")})
    gap()

    st.subheader("Method")
    st.markdown(
        "- **Task 1, classification:** will a station reach alert level or above within 24 hours?\n"
        "- **Task 2, regression:** water level 24 hours ahead.\n"
        "- **Models:** bagging (Random Forest, Extra Trees), boosting (AdaBoost, XGBoost, LightGBM, CatBoost), "
        "stacking, and two hybrids (RF + ANN, RF + ARIMA).\n"
        "- **Evaluation:** chronological train / validation / test split. Hyperparameters and thresholds tuned on "
        "validation, test set used once. Persistence (no change) is the baseline.")


def classification():
    page_header("Model Comparison", "Classification: alert level or above within the next 24 hours.")
    split = split_choice("clf_split")
    t = D["clf_test"] if split == "test" else D["clf_val"]
    fams = st.pills("Model families", list(FAM_COLORS)[:1] + ["Bagging", "Boosting", "Stacking"],
                    default=["Baseline", "Bagging", "Boosting", "Stacking"], selection_mode="multi")
    t = t[t.Family.isin(fams or [])]
    if t.empty:
        st.info("Select at least one model family.")
        return

    st.subheader("Best model per family")
    bf = t.sort_values("PR_AUC", ascending=False).groupby("Family").head(1)
    cols = st.columns(len(bf))
    for col, (_, r) in zip(cols, bf.iterrows()):
        col.metric(f"{r.Family}: {r.Model}", f"{r.PR_AUC:.3f}", f"F1 {r.F1:.3f}", delta_color="off")

    family_key(sorted(t.Family.unique()))
    c1, c2 = st.columns(2)
    c1.plotly_chart(bar_chart(t, "PR_AUC", "PR-AUC (higher is better)"))
    c2.plotly_chart(bar_chart(t, "F1", "F1 score"))
    st.caption("PR-AUC used as the main metric because only about 10% of readings are positive.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("95% confidence intervals (test)")
        ci = D["clf_ci"].copy()
        lo_hi = ci.PR_AUC_95CI.str.strip("[]").str.split(",", expand=True).astype(float)
        ci["lo"], ci["hi"] = lo_hi[0], lo_hi[1]
        ci = ci.merge(D["clf_test"][["Model", "Family"]], on="Model").sort_values("PR_AUC")
        ci = ci[ci.Family.isin(fams)]
        fig = go.Figure(go.Scatter(x=ci.PR_AUC, y=ci.Model, mode="markers",
                                   marker=dict(size=10, color=ci.Family.map(FAM_COLORS)),
                                   error_x=dict(type="data", symmetric=False, array=ci.hi - ci.PR_AUC,
                                                arrayminus=ci.PR_AUC - ci.lo, color="#888")))
        fig.update_layout(height=60 + 28 * len(ci), margin=dict(t=10, b=10), xaxis_title="test PR-AUC")
        st.plotly_chart(fig)
        st.caption("Bootstrap, 1,000 resamples.")
    with c2:
        st.subheader("New alert detection")
        o = D["clf_onset"][D["clf_onset"].Family.isin(fams)]
        st.plotly_chart(bar_chart(o, "Recall", "Share of new alerts caught"))
        st.caption("Recall on stations currently Normal that reach alert within 24 h.")

    with st.expander("Full results table"):
        st.dataframe(fmt_table(t.drop(columns=["Fit_s"]), ["PR_AUC", "ROC_AUC", "F1", "Recall", "Precision", "MCC",
                                                             "Bal_Acc", "Accuracy", "Threshold"]),
                     width="stretch", hide_index=True)


def monitor():
    page_header("Station View", "Observed water level and predicted alert probability, test period.")
    P, C = D["clf_pred"], D["clean"]
    models = [c.replace("proba__", "") for c in P.columns if c.startswith("proba__")]
    busiest = P.groupby("station").y_true.sum().idxmax()
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        basin = station_picker("River basin", P.basin.unique(), default=P.loc[P.station == busiest, "basin"].iloc[0],
                               key="mon_basin")
    with c2:
        station = station_picker("Station", P[P.basin == basin].station.unique(), default=busiest, key="mon_station")
    chosen = c3.multiselect("Models", models, default=[m for m in ["CatBoost", "Random Forest", "Soft Voting"] if m in models])

    p = P[P.station == station].sort_values("obs_datetime")
    h = C[(C.station == station) & (C.obs_datetime >= CUT_TEST - pd.Timedelta(days=7))].sort_values("obs_datetime")
    last = h.iloc[-1]
    k = st.columns(4)
    k[0].metric("Latest level", f"{last.water_level:.2f} m")
    k[1].metric("Alert level", f"{last.alert_level:.2f} m")
    k[2].metric("Status", STATUS.get(int(last.status), "Unknown") if pd.notna(last.status) else "Unknown")
    k[3].metric("Alert periods in test", int(p.y_true.sum()))

    fig = go.Figure()
    for col, name, colr in [("major_flood_level", "Major flood", "#C9504A"), ("minor_flood_level", "Minor flood", "#E08A3C"),
                            ("alert_level", "Alert", "#E3B23C")]:
        fig.add_scatter(x=h.obs_datetime, y=h[col], mode="lines", name=name, line=dict(color=colr, width=1.2))
    fig.add_scatter(x=h.obs_datetime, y=h.water_level, mode="lines+markers", name="Water level",
                    line=dict(color="#3D7EBF", width=2), marker=dict(size=3))
    fig.update_layout(title=dict(text="Water level and official thresholds", font=dict(size=15)), height=320,
                      yaxis_title="metres", margin=dict(t=45, b=10), legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(tickformat="%d %b"))
    st.plotly_chart(fig)

    fig = go.Figure()
    pos = p[p.y_true == 1]
    fig.add_scatter(x=pos.obs_datetime, y=[1.04] * len(pos), mode="markers", name="Alert actually followed",
                    marker=dict(symbol="square", color="#C9504A", size=6))
    for m in chosen:
        fam = D["clf_reg"].get(m, {}).get("family", "Baseline")
        fig.add_scatter(x=p.obs_datetime, y=p[f"proba__{m}"], mode="lines", name=m,
                        line=dict(color=FAM_COLORS.get(fam), width=1.6))
    if len(chosen) == 1:
        fig.add_hline(y=D["clf_reg"][chosen[0]]["threshold"], line_width=1, line_color="#888",
                      annotation_text="warning threshold")
    fig.update_layout(title=dict(text="Predicted chance of an alert within 24 hours", font=dict(size=15)),
                      height=320, yaxis=dict(range=[0, 1.09], title="probability"), margin=dict(t=45, b=10),
                      legend=dict(orientation="h", y=-0.15), xaxis=dict(tickformat="%d %b"))
    st.plotly_chart(fig)

    if chosen:
        rows = []
        for m in chosen:
            thr = D["clf_reg"][m]["threshold"]
            pred = (p[f"proba__{m}"] >= thr).astype(int)
            rows.append({"Model": m, "Warnings issued": int(pred.sum()),
                         "Correct warnings": int(((pred == 1) & (p.y_true == 1)).sum()),
                         "Missed alerts": int(((pred == 0) & (p.y_true == 1)).sum()),
                         "False alarms": int(((pred == 1) & (p.y_true == 0)).sum())})
        st.subheader("Warnings at this station")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def regression():
    page_header("Water Level Forecast", "Regression: water level 24 hours ahead, including the RF + ANN hybrids.")
    split = split_choice("reg_split")
    t = D["reg_test"] if split == "test" else D["reg_val"]
    family_key(sorted(t.Family.unique()))
    c1, c2 = st.columns(2)
    c1.plotly_chart(bar_chart(t, "Skill", "Skill vs persistence", fmt="{:.1%}"))
    c2.plotly_chart(bar_chart(t, "MAE_high", "MAE at high water (m)", ascending=False))
    st.caption("Skill = 1 - RMSE / RMSE(persistence). High water = true level at least 80% of the alert level.")

    st.subheader("RF + ANN hybrids")
    st.markdown("- **H1 residual:** RF forecast + ANN trained on the RF's out-of-fold errors\n"
                "- **H2 stacking:** RF and ANN combined by a non-negative linear meta-model\n"
                "- **H3 blend:** weighted average w x RF + (1 - w) x ANN, w tuned on validation")

    st.subheader("Error by water level band")
    b = D["reg_band"].set_index("Water level band")
    bm = b.drop(columns="rows").reset_index().melt(id_vars="Water level band", var_name="Model", value_name="MAE")
    bm["Water level band"] = bm["Water level band"].replace({
        "<50% alert": "Below half of alert", "50-80%": "50 to 80% of alert",
        "80-100%": "80 to 100% of alert", ">= alert": "At or above alert"})
    fam_of = D["reg_test"].set_index("Model").Family
    fig = px.bar(bm, x="Water level band", y="MAE", color="Model", barmode="group", height=340,
                 color_discrete_map={m: FAM_COLORS.get(fam_of.get(m, "Baseline")) for m in bm.Model.unique()})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(margin=dict(t=10, b=10), yaxis_title="average error (m)", xaxis_title=None,
                      legend=dict(orientation="h", y=-0.15, title=None))
    st.plotly_chart(fig)
    st.caption("Test set MAE, grouped by the observed level as a share of the station's alert level.")

    st.subheader("Forecast vs observed")
    R = D["reg_pred"]
    models = [c.replace("pred__", "") for c in R.columns if c.startswith("pred__")]
    c1, c2 = st.columns([1, 2])
    with c1:
        station = station_picker("Station", R.station.unique(), key="reg_station")
    chosen = c2.multiselect("Models", models, default=[m for m in ["Persistence", "Random Forest", "H3 RF+ANN blend"]
                                                       if m in models], key="reg_models")
    r = R[R.station == station].sort_values("obs_datetime")
    tt = r.obs_datetime + pd.Timedelta(hours=24)
    fig = go.Figure()
    fig.add_scatter(x=tt, y=r.y_true, mode="lines+markers", name="Observed", line=dict(color="#D0D4DA", width=2),
                    marker=dict(size=3))
    fams = D["reg_test"].set_index("Model").Family
    for m in chosen:
        fig.add_scatter(x=tt, y=r[f"pred__{m}"], mode="lines", name=m,
                        line=dict(color=FAM_COLORS.get(fams.get(m, "Baseline")), width=1.5))
    fig.add_hline(y=float(r.alert_level.iloc[0]), line_width=1, line_color="#E3B23C", annotation_text="alert level")
    fig.update_layout(height=360, yaxis_title="metres", margin=dict(t=20, b=10), legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(tickformat="%d %b"))
    st.plotly_chart(fig)

    with st.expander("Full results table"):
        st.dataframe(fmt_table(t.drop(columns=["Fit_s"]), ["MAE", "RMSE", "R2", "Skill", "MAE_high", "Warn_F1",
                                                             "Warn_Recall"]), width="stretch", hide_index=True)


def timeseries():
    page_header("RF + ARIMA Hybrid", "ARIMA forecast per station, corrected by a Random Forest trained on rainfall and basin features.")
    split = split_choice("ts_split")
    t = (D["ts_test"] if split == "test" else D["ts_val"]).copy()
    k = st.columns(len(t))
    for col, (_, r) in zip(k, t.iterrows()):
        col.metric(f"{r.Model} RMSE", f"{r.RMSE:.3f} m", f"{r.Skill:.1%} skill" if r.Skill > 0 else None)
    ps = D["ts_station"]
    wins = (ps["RF + ARIMA hybrid"] < ps["ARIMA"]).mean()
    st.caption(f"Hybrid better than ARIMA at {wins:.0%} of {len(ps)} stations (test RMSE).")

    TP = D["ts_pred"]
    c1, c2 = st.columns([1, 2])
    with c1:
        station = station_picker("Station", TP.station.unique(), key="ts_station")
        o = D["ts_orders"].set_index("station")
        if station in o.index:
            st.caption(f"Model used: ARIMA{o.loc[station, 'order']}")
    tp = TP[TP.station == station].sort_values("target_time")
    fig = go.Figure()
    fig.add_scatter(x=tp.target_time, y=tp.actual_24h, mode="lines+markers", name="Observed",
                    line=dict(color="#D0D4DA", width=2), marker=dict(size=3))
    for m in ["ARIMA", "RF (direct)", "RF + ARIMA hybrid"]:
        fig.add_scatter(x=tp.target_time, y=tp[f"pred__{m}"], mode="lines", name=m, line=dict(color=TS_COLORS[m], width=1.5))
    fig.add_hline(y=float(tp.alert_level.iloc[0]), line_width=1, line_color="#E3B23C", annotation_text="alert level")
    fig.update_layout(height=380, yaxis_title="metres", title=dict(text="24-hour forecasts", font=dict(size=15)),
                      margin=dict(t=45, b=10), legend=dict(orientation="h", y=-0.15), xaxis=dict(tickformat="%d %b"))
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    for col, metric, title, fmt in [(c1, "Skill", "Error removed compared with persistence", "{:.1%}"),
                                    (c2, "MAE_high", "Average error near flood level (m)", "{:.3f}")]:
        fig = go.Figure(go.Bar(x=t.Model, y=t[metric], marker_color=[TS_COLORS[m] for m in t.Model],
                               text=[fmt.format(v) for v in t[metric]], textposition="outside"))
        fig.update_layout(title=dict(text=title, font=dict(size=15)), height=320, margin=dict(t=45, b=10))
        col.plotly_chart(fig)
    with st.expander("Error at every station"):
        st.dataframe(ps.round(3), width="stretch", hide_index=True)


def explain():
    page_header("Feature Importance", "Normalised importance from the tree-based models.")
    I = D["imp"]
    c1, c2, c3 = st.columns([1, 2, 1])
    task = c1.segmented_control("Task", ["Flood warning", "Level forecast"], default="Flood warning")
    tk = "reg" if task == "Level forecast" else "clf"
    models = sorted(I[I.task == tk].model.unique())
    chosen = c2.multiselect("Models", models, default=models)
    topn = c3.slider("Features shown", 5, 25, 12)
    sub = I[(I.task == tk) & (I.model.isin(chosen))]
    if sub.empty:
        st.info("Select at least one model.")
        return
    order = sub.groupby("feature").importance.mean().sort_values(ascending=False).head(topn).index
    sub = sub[sub.feature.isin(order)]
    fig = px.bar(sub, x="importance", y="feature", color="model", barmode="group", orientation="h",
                 category_orders={"feature": list(order)}, height=140 + 36 * topn)
    fig.update_layout(margin=dict(t=10, b=10), xaxis_title="relative importance", yaxis_title=None,
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig)
    st.caption("Threshold-relative features (level / alert level, margins) rank highest, followed by rainfall.")


def whatif():
    page_header("What-if Analysis", "Pick a test reading, change the inputs and see what a trained model predicts.")
    use_trained = tm.available()
    X, ids = D["X_test"].reset_index(drop=True), D["ids_test"].reset_index(drop=True)

    m1, m2 = st.columns(2)
    if use_trained:
        clf_names, reg_names = tm.clf_choices(), tm.reg_choices()
        clf_name = m1.selectbox("Flood warning model", clf_names, index=clf_names.index("CatBoost") if "CatBoost" in clf_names else 0)
        reg_name = m2.selectbox("Water level model", reg_names, index=reg_names.index("CatBoost") if "CatBoost" in reg_names else 0,
                                help="XGBoost-based regressors are left out: they were saved with XGBoost 3 in Colab.")
        feats = tm.preprocessors()[1]["features"]
    else:
        try:
            clf_m, reg_m = load_models()
        except Exception as e:
            st.error(f"Could not load the CatBoost models ({e}). Install them with `pip install catboost`.")
            return
        clf_name = reg_name = "CatBoost"
        feats = clf_m.feature_names_
        m1.caption("Using the CatBoost models from the bundle. Add the trained model folders to dashboard/models to choose others.")
    thr = D["clf_reg"].get(clf_name, {}).get("threshold", 0.5)
    w_rf = float(tm.specs("reg").get("H3 RF+ANN blend", {}).get("params", {}).get("w_rf", 0.65)) if use_trained else 0.65

    def p_alert(frame):
        return tm.predict_proba(clf_name, frame) if use_trained else clf_m.predict_proba(frame[feats])[:, 1]

    def change(frame):
        return tm.predict_change(reg_name, frame, w_rf) if use_trained else reg_m.predict(frame[feats])

    c1, c2 = st.columns(2)
    with c1:
        station = station_picker("Station", ids.station.unique(), key="wi_station")
    rows = ids.index[ids.station == station]
    times = pd.to_datetime(ids.loc[rows, "obs_datetime"]).dt.strftime("%d %b %Y, %H:%M").tolist()
    default_i = int(np.argmax(X.loc[rows, "water_level"].values))
    t_sel = c2.selectbox("Starting reading", times, index=default_i, help="Defaults to the highest level in the test period.")
    i0 = rows[times.index(t_sel)]
    base = X.loc[i0, feats].astype(float).copy()
    al, mi, ma = base["alert_level"], base["minor_flood_level"], base["major_flood_level"]

    with st.container(border=True):
        st.markdown(f"Inputs. Thresholds: alert {al:.2f} m, minor {mi:.2f} m, major {ma:.2f} m")
        s1, s2, s3 = st.columns(3)
        lvl = s1.slider("Water level (m)", float(min(base.water_level, 0) - 1), float(max(ma * 1.3, base.water_level + 1)),
                        float(base.water_level), 0.05)
        rise = s2.slider("1-hour change (m)", -1.0, 1.0, float(np.nan_to_num(base.level_change_in_report)), 0.01)
        r24 = s3.slider("Rainfall 24 h (mm)", 0.0, 300.0, float(np.nan_to_num(base.rainfall_24h_mm)), 1.0)
        b1, b2 = st.columns(2)
        r3 = b1.slider("Rainfall 3 days (mm)", 0.0, 600.0, float(max(np.nan_to_num(base.rain_3d_sum_mm), r24)), 1.0)
        brain = b2.slider("Basin mean rainfall 24 h (mm)", 0.0, 300.0, float(np.nan_to_num(base.basin_rain_mean_mm)), 1.0)

    x = base.copy()
    x["water_level"], x["level_prev_in_report"], x["level_change_in_report"] = lvl, lvl - rise, rise
    x["rainfall_24h_mm"], x["rain_3d_sum_mm"] = r24, max(r3, r24)
    x["rain_7d_sum_mm"] = max(np.nan_to_num(base.rain_7d_sum_mm), x["rain_3d_sum_mm"])
    x["basin_rain_mean_mm"], x["basin_rain_max_mm"] = brain, max(np.nan_to_num(base.basin_rain_max_mm), brain, r24)
    x["level_to_alert_ratio"], x["level_to_major_ratio"] = lvl / al, lvl / ma
    x["margin_to_alert_m"], x["margin_to_minor_m"], x["margin_to_major_m"] = al - lvl, mi - lvl, ma - lvl
    x["status"] = 3 if lvl >= ma else 2 if lvl >= mi else 1 if lvl >= al else 0
    x["trend_num"] = 1.0 if rise > 0.005 else -1.0 if rise < -0.005 else 0.0
    if "rain_intensity_3d" in x:
        x["rain_intensity_3d"] = x["rain_3d_sum_mm"] / 3
    if "basin_max_alert_ratio" in x:
        x["basin_max_alert_ratio"] = max(np.nan_to_num(base.basin_max_alert_ratio), lvl / al)

    frame = pd.DataFrame([base.values, x.values], columns=feats)
    proba = p_alert(frame)
    lvl24 = frame["water_level"].values + change(frame)
    warn = proba[1] >= thr
    status24 = STATUS[3 if lvl24[1] >= ma else 2 if lvl24[1] >= mi else 1 if lvl24[1] >= al else 0]

    g1, g2 = st.columns([1.2, 1])
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=proba[1] * 100,
                                     number={"suffix": "%", "valueformat": ".0f"},
                                     title={"text": f"P(alert within 24 h), {clf_name}", "font": {"size": 15}},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#C9504A" if warn else "#4E9F6E"},
                                            "threshold": {"line": {"color": "#555", "width": 3}, "value": thr * 100}}))
        fig.update_layout(height=270, margin=dict(t=60, b=10))
        st.plotly_chart(fig)
    with g2:
        if warn:
            st.error(f"Warning: probability {proba[1]:.0%} is above the threshold ({thr:.0%}).")
        else:
            st.success(f"No warning: probability {proba[1]:.0%} is below the threshold ({thr:.0%}).")
        st.metric(f"Forecast level (24 h), {reg_name}", f"{lvl24[1]:.2f} m", f"{lvl24[1] - lvl:+.2f} m", delta_color="inverse")
        st.write(f"Current status: {STATUS[int(x['status'])]}  \nForecast status: {status24}")
        st.caption(f"Original reading: {proba[0]:.0%}")

    with st.expander("Sensitivity to water level"):
        grid = np.linspace(min(0, lvl) - 0.5, max(ma * 1.2, lvl + 0.5), 60)
        G = pd.DataFrame([x.values] * len(grid), columns=feats)
        G["water_level"] = grid
        G["level_prev_in_report"] = grid - rise
        G["level_to_alert_ratio"], G["level_to_major_ratio"] = grid / al, grid / ma
        G["margin_to_alert_m"], G["margin_to_minor_m"], G["margin_to_major_m"] = al - grid, mi - grid, ma - grid
        G["status"] = np.select([grid >= ma, grid >= mi, grid >= al], [3, 2, 1], 0)
        fig = go.Figure(go.Scatter(x=grid, y=p_alert(G), mode="lines", line=dict(color="#C9504A", width=2)))
        for v, n, c in [(al, "alert", "#E3B23C"), (mi, "minor", "#E08A3C"), (ma, "major", "#C9504A")]:
            fig.add_vline(x=v, line_width=1, line_color=c, annotation_text=n)
        fig.add_hline(y=thr, line_width=1, line_color="#888", annotation_text="warning threshold")
        fig.update_layout(height=300, xaxis_title="water level (m)", yaxis_title="P(alert)", margin=dict(t=20, b=10))
        st.plotly_chart(fig)
        st.caption("Other inputs fixed at the values above.")


def trained_models():
    page_header("Trained Models", "The saved models from Colab, loaded and scored on the test set.")
    if not tm.available():
        st.info("Trained model folders not found. Unzip models_clf, models_reg and models_ts into `dashboard/models/`.")
        return
    from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, recall_score, precision_score,
                                 mean_absolute_error, mean_squared_error, confusion_matrix)
    tab_c, tab_r, tab_t = st.tabs(["Classification", "Regression", "RF + ARIMA"])

    with tab_c:
        inv = tm.inventory("clf")
        inv["Threshold"] = inv.Model.map(lambda m: round(D["clf_reg"].get(m, {}).get("threshold", np.nan), 3))
        st.dataframe(inv, hide_index=True, width="stretch")
        name = st.selectbox("Model", inv.Model.tolist(), index=inv.Model.tolist().index("CatBoost"), key="tm_clf")
        if st.button("Evaluate on test set", key="tm_clf_btn", type="primary"):
            Xr, y = tm.test_inputs("clf")
            p = tm.predict_proba(name, Xr)
            thr = D["clf_reg"][name]["threshold"]
            pred = (p >= thr).astype(int)
            got = {"PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
                   "F1": f1_score(y, pred), "Recall": recall_score(y, pred), "Precision": precision_score(y, pred)}
            saved = D["clf_test"].set_index("Model").loc[name]
            diff = max(abs(got[k] - saved[k]) for k in got)
            k = st.columns(5)
            for col, (m, v) in zip(k, got.items()):
                col.metric(m.replace("_", "-"), f"{v:.4f}")
            if diff < 1e-6:
                st.success("Matches the saved test result from the notebook.")
            else:
                st.warning(f"Differs from the saved result by up to {diff:.4f}.")
            c1, c2 = st.columns([1, 1.4])
            cm = confusion_matrix(y, pred)
            c1.dataframe(pd.DataFrame(cm, index=["Actual: no alert", "Actual: alert"],
                                      columns=["Predicted: no alert", "Predicted: alert"]), width="stretch")
            ids = D["ids_test"].reset_index(drop=True)
            out = pd.DataFrame({"obs_datetime": ids.obs_datetime, "station": ids.station, "basin": ids.basin,
                                "actual_alert_24h": y, "probability": p.round(4), "warning": pred})
            c2.download_button("Download test predictions (CSV)", out.to_csv(index=False), f"{tm.fname(name)[:-7]}_test_predictions.csv",
                               "text/csv")

    with tab_r:
        inv = tm.inventory("reg")
        st.dataframe(inv, hide_index=True, width="stretch")
        st.caption("Models marked with XGBoost (XGBoost, Averaging, Stacking) were saved with XGBoost 3 in Colab; "
                   "the local XGBoost 2 reads them differently, so their live scores will not match.")
        names = inv.Model.tolist() + (["H3 RF+ANN blend"] if "H3 RF+ANN blend" in tm.specs("reg") else [])
        name = st.selectbox("Model", names, index=names.index("Random Forest"), key="tm_reg")
        if st.button("Evaluate on test set", key="tm_reg_btn", type="primary"):
            Xr, y = tm.test_inputs("reg")
            w_rf = float(tm.specs("reg").get("H3 RF+ANN blend", {}).get("params", {}).get("w_rf", 0.65))
            now = Xr["water_level"].values
            pred = now + tm.predict_change(name, Xr, w_rf)
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            got = {"RMSE": rmse, "MAE": mean_absolute_error(y, pred),
                   "Skill": 1 - rmse / float(np.sqrt(mean_squared_error(y, now)))}
            saved = D["reg_test"].set_index("Model").loc[name]
            diff = max(abs(got[k] - saved[k]) for k in got)
            k = st.columns(3)
            k[0].metric("RMSE (m)", f"{got['RMSE']:.4f}"); k[1].metric("MAE (m)", f"{got['MAE']:.4f}")
            k[2].metric("Skill vs persistence", f"{got['Skill']:.1%}")
            if diff < 1e-6:
                st.success("Matches the saved test result from the notebook.")
            elif name in tm.XGB_REGRESSION:
                st.warning(f"Differs from the saved result by {diff:.4f}: this model contains XGBoost and needs XGBoost 3.x "
                           "(the Colab version) to load exactly. Use the notebook result for this model.")
            else:
                st.warning(f"Differs from the saved result by up to {diff:.4f}.")
            fig = go.Figure(go.Scattergl(x=y, y=pred, mode="markers", marker=dict(size=3, opacity=0.4, color="#3D7EBF")))
            lim = [float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))]
            fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="#888", width=1), showlegend=False)
            fig.update_layout(height=340, xaxis_title="observed level in 24 h (m)", yaxis_title="predicted (m)",
                              margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig)

    with tab_t:
        pack = tm.load("ts", "rf_arima_hybrid")
        orders = pd.DataFrame([{"Station": s, "ARIMA order": str(tuple(o)), "Trend": pack["arima_trend"][s]}
                               for s, o in pack["arima_orders"].items()])
        k = st.columns(3)
        k[0].metric("Stations with an ARIMA model", len(orders))
        k[1].metric("Hybrid RF inputs", len(pack["x_cols"]))
        k[2].metric("Trees in hybrid RF", pack["rf_hybrid"].n_estimators)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(orders, hide_index=True, width="stretch", height=300)
        with c2:
            imp = pd.Series(pack["rf_hybrid"].feature_importances_, index=pack["x_cols"]).sort_values(ascending=False).head(10)
            fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h", marker_color="#C9504A"))
            fig.update_layout(title=dict(text="What the hybrid RF corrects in ARIMA (top 10)", font=dict(size=14)),
                              height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig)
        st.subheader("Rebuild one station's forecasts")
        TP = D["ts_pred"]
        station = station_picker("Station", TP.station.unique(), key="tm_ts_station")
        if st.button("Rebuild with the saved ARIMA parameters", key="tm_ts_btn", type="primary"):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            meta = tm.preprocessors()[1]
            clean = D["clean"]
            grid = pd.date_range(clean.obs_datetime.min().floor("3h"), clean.obs_datetime.max(), freq="3h")
            s = clean[clean.station == station].drop_duplicates("obs_datetime").set_index("obs_datetime")["water_level"].reindex(grid)
            res = SARIMAX(s, order=pack["arima_orders"][station], trend=pack["arima_trend"][station]).filter(
                pack["arima_params"][station])
            ref = TP[TP.station == station].sort_values("obs_datetime")
            pos = {t: i for i, t in enumerate(grid)}
            fc = np.array([res.get_prediction(start=pos[t] + 1, end=pos[t] + 8, dynamic=0).predicted_mean.iloc[-1]
                           for t in ref.obs_datetime])
            feats = clean[clean.station == station].set_index("obs_datetime").loc[ref.obs_datetime, meta["features"]]
            feats = feats.assign(arima_pred=fc, arima_change=fc - ref.level_now.values)
            Xi = pd.DataFrame(pack["imputer"].transform(feats[pack["x_cols"]]), columns=pack["x_cols"])
            hyb = fc + pack["rf_hybrid"].predict(Xi)
            d1 = float(np.abs(fc - ref["pred__ARIMA"].values).max())
            d2 = float(np.abs(hyb - ref["pred__RF + ARIMA hybrid"].values).max())
            (st.success if max(d1, d2) < 1e-6 else st.warning)(
                f"{len(ref)} test forecasts rebuilt. Largest difference from the saved forecasts: ARIMA {d1:.2e} m, hybrid {d2:.2e} m.")
            fig = go.Figure()
            fig.add_scatter(x=ref.target_time, y=ref.actual_24h, mode="lines+markers", name="Observed",
                            line=dict(color="#D0D4DA", width=2), marker=dict(size=3))
            fig.add_scatter(x=ref.target_time, y=fc, mode="lines", name="ARIMA (rebuilt)", line=dict(color="#8E7CC3"))
            fig.add_scatter(x=ref.target_time, y=hyb, mode="lines", name="RF + ARIMA (rebuilt)", line=dict(color="#C9504A"))
            fig.update_layout(height=340, yaxis_title="metres", margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.15),
                              xaxis=dict(tickformat="%d %b"))
            st.plotly_chart(fig)

pages = {
    "Home": [st.Page(overview, title="Overview", icon=":material/home:", default=True)],
    "Classification": [
        st.Page(classification, title="Model comparison", icon=":material/leaderboard:", url_path="comparison"),
        st.Page(monitor, title="Station view", icon=":material/location_on:", url_path="monitor"),
    ],
    "Forecasting": [
        st.Page(regression, title="Level forecast", icon=":material/show_chart:", url_path="forecast"),
        st.Page(timeseries, title="RF + ARIMA", icon=":material/timeline:", url_path="arima"),
    ],
    "Analysis": [
        st.Page(explain, title="Feature importance", icon=":material/insights:", url_path="features"),
        st.Page(whatif, title="What-if analysis", icon=":material/tune:", url_path="scenario"),
        st.Page(trained_models, title="Trained models", icon=":material/model_training:", url_path="models"),
    ],
}

st.logo(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.svg"), size="large")
nav = st.navigation(pages)

with st.sidebar:
    with st.expander("About the data", expanded=False):
        st.markdown(
            f"Water level and rainfall reports from the Irrigation Department of Sri Lanka, "
            f"{D['clean'].obs_datetime.min():%B %Y} to {END:%B %Y}.\n\n"
            f"Training data runs to {CUT_VAL - pd.Timedelta(hours=3):%d %b %Y}. "
            f"Models were tuned on {CUT_VAL:%d %b} to {CUT_TEST - pd.Timedelta(hours=3):%d %b %Y} "
            f"and tested on {CUT_TEST:%d %b} to {END:%d %b %Y}.")

nav.run()
