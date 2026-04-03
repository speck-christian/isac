"""Streamlit dashboard for the portfolio benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from isac.analysis import evaluate_selectors, make_portfolio_table
from isac.core import PortfolioBenchmark

PRECOMPUTED_SWEEP_PATH = Path("artifacts/noise_sweep/dashboard_curves.csv")

REGIME_COLOR_MAP = {
    "Regime 0": "#0f6e76",
    "Regime 1": "#d8892b",
    "Regime 2": "#b44c32",
    "Regime 3": "#6e7f52",
    "Regime 4": "#655e8f",
}

st.set_page_config(
    page_title="ISAC Portfolio Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(246, 181, 70, 0.20), transparent 32%),
            radial-gradient(circle at top right, rgba(15, 110, 118, 0.22), transparent 28%),
            linear-gradient(180deg, #f6f2e8 0%, #efe8da 100%);
        color: #1f2a2e;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 0.6rem;
        max-width: 1400px;
    }
    .hero-card {
        background: rgba(255, 252, 246, 0.88);
        border: 1px solid rgba(31, 42, 46, 0.08);
        border-radius: 22px;
        box-shadow: 0 18px 50px rgba(60, 53, 35, 0.10);
        backdrop-filter: blur(6px);
        padding: 1.4rem 1.6rem;
        margin-bottom: 0.3rem;
    }
    .eyebrow {
        color: #a75a1d;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .hero-title {
        font-size: 2.4rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0.2rem 0 0.4rem 0;
        color: #122126;
    }
    .hero-copy {
        color: #415157;
        font-size: 1rem;
        max-width: 56rem;
        margin-bottom: 0;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #122126;
        margin-bottom: 0.08rem;
    }
    .control-card {
        background: rgba(255, 252, 246, 0.82);
        border: 1px solid rgba(31, 42, 46, 0.08);
        border-radius: 18px;
        padding: 0.8rem 1rem 0.2rem 1rem;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def build_dashboard_data(
    n_instances: int,
    feature_noise: float,
    parameter_noise: float,
    runtime_noise: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = PortfolioBenchmark(
        feature_noise=feature_noise,
        parameter_noise=parameter_noise,
        runtime_noise=runtime_noise,
        seed=seed,
    )
    selector_table, action_table, instance_table = evaluate_selectors(
        benchmark=benchmark,
        n_instances=n_instances,
        seed=seed,
    )
    portfolio_table = make_portfolio_table(benchmark)
    return selector_table, action_table, instance_table, portfolio_table


@st.cache_data(show_spinner=False)
def load_precomputed_noise_sweep_data() -> pd.DataFrame:
    return pd.read_csv(PRECOMPUTED_SWEEP_PATH)


def plot_parameter_surface(
    frame: pd.DataFrame,
    *,
    color_column: str,
    title: str,
    selector_name: str | None = None,
) -> object:
    hover_fields = {
        "instance_id": True,
        "regime_label": True,
        "best_config": True,
        "x": False,
        "y": False,
    }
    if selector_name is not None:
        selector_key = selector_name.lower().replace(" ", "_")
        hover_fields[f"choice_name_{selector_key}"] = True
        hover_fields[f"regret_{selector_key}"] = ":.3f"

    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color=color_column,
        symbol="regime_label",
        hover_data=hover_fields,
        color_continuous_scale=["#0f6e76", "#efe8da", "#b44c32"],
        labels={
            "x": "Embedding Axis 1",
            "y": "Embedding Axis 2",
            color_column: title,
            "regime_label": "Regime",
        },
    )
    figure.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.4, color="#f7f3ea")))
    figure.update_layout(
        height=360,
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    )
    return figure


sweep_controls = st.columns((1.05, 1.0, 0.95), gap="small")
with sweep_controls[0]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown("**Sweep protocol**")
    st.caption("Fixed robustness sweep with 360 train/test instances per setting.")
    st.markdown("</div>", unsafe_allow_html=True)
with sweep_controls[1]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown("**Seed averaging**")
    st.caption("Curves are averaged over 8 random seeds for each noise setting.")
    st.markdown("</div>", unsafe_allow_html=True)
with sweep_controls[2]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    curve_metric = st.selectbox(
        "Curve metric",
        options=["avg_regret", "avg_runtime", "accuracy"],
        index=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)

seed = 7
n_instances = 360
feature_noise = 0.40
parameter_noise = 0.08
runtime_noise = 0.03
selected_selector = "DGCAC-inspired"
sweep_instances = 360
sweep_seed_count = 8

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Instance-Specific Portfolio Selection</div>
        <div class="hero-title">Core benchmark views</div>
        <p class="hero-copy">
            This dashboard focuses on the four views that matter most:
            true parameter structure, selected parameter structure, average algorithm
            performance, and robustness curves over the three noise variables.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Algorithm Curves vs Noise Variables</div>',
    unsafe_allow_html=True,
)
sweep_results = load_precomputed_noise_sweep_data()
sweep_results = sweep_results[
    sweep_results["selector"].isin(
        [
            "DGCAC-inspired",
            "Cluster ISAC",
            "Classifier",
            "MLP Selector",
            "Regressor",
        ]
    )
].copy()

noise_columns = ["feature_noise", "parameter_noise", "runtime_noise"]
subplot_titles = ["Feature Noise", "Parameter Noise", "Runtime Noise"]
curve_figure = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=subplot_titles,
    shared_yaxes=True,
    horizontal_spacing=0.04,
)

selector_order = [
    "DGCAC-inspired",
    "Cluster ISAC",
    "Classifier",
    "MLP Selector",
    "Regressor",
]
selector_colors = {
    "DGCAC-inspired": "#0f6e76",
    "Cluster ISAC": "#d8892b",
    "Classifier": "#b44c32",
    "MLP Selector": "#8c5e3c",
    "Regressor": "#6e7f52",
}

for col_index, noise_name in enumerate(noise_columns, start=1):
    curve_table = (
        sweep_results.groupby(["selector", noise_name], as_index=False)[curve_metric].mean()
    )
    for selector_name in selector_order:
        selector_slice = curve_table[curve_table["selector"] == selector_name]
        curve_figure.add_trace(
            go.Scatter(
                x=selector_slice[noise_name],
                y=selector_slice[curve_metric],
                mode="lines+markers",
                name=selector_name,
                legendgroup=selector_name,
                showlegend=col_index == 1,
                line=dict(width=3, color=selector_colors[selector_name]),
                marker=dict(size=9, color=selector_colors[selector_name]),
            ),
            row=1,
            col=col_index,
        )
    curve_figure.update_xaxes(title_text=noise_name.replace("_", " "), row=1, col=col_index)

curve_figure.update_yaxes(title_text=curve_metric.replace("_", " "), row=1, col=1)
curve_figure.update_layout(
    height=460,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=36, b=10),
    font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    legend_title_text="",
)
st.plotly_chart(curve_figure, use_container_width=True)

distribution_selector = st.selectbox(
    "Distribution algorithm",
    options=selector_order,
    index=0,
)
distribution_table = (
    sweep_results[sweep_results["selector"] == distribution_selector]
    .groupby(["feature_noise", "parameter_noise", "runtime_noise"], as_index=False)[curve_metric]
    .mean()
)
distribution_chart = px.scatter_3d(
    distribution_table,
    x="feature_noise",
    y="parameter_noise",
    z="runtime_noise",
    color=curve_metric,
    size=curve_metric,
    color_continuous_scale=["#0f6e76", "#efe8da", "#b44c32"],
    labels={
        "feature_noise": "Feature noise",
        "parameter_noise": "Parameter noise",
        "runtime_noise": "Runtime noise",
        curve_metric: curve_metric.replace("_", " "),
    },
    hover_data={
        "feature_noise": ":.2f",
        "parameter_noise": ":.2f",
        "runtime_noise": ":.2f",
        curve_metric: ":.3f",
    },
)
distribution_chart.update_traces(marker=dict(opacity=0.82, line=dict(width=0.3, color="#fffaf0")))
distribution_chart.update_layout(
    title="Performance across the full three-noise grid",
    height=520,
    paper_bgcolor="rgba(255,255,255,0.0)",
    margin=dict(l=10, r=10, t=42, b=10),
    font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    scene=dict(
        xaxis_title="Feature noise",
        yaxis_title="Parameter noise",
        zaxis_title="Runtime noise",
        bgcolor="rgba(255,252,246,0.55)",
    ),
)
st.plotly_chart(distribution_chart, use_container_width=True)

st.markdown(
    '<div class="section-title">Interpretable Robustness Summaries</div>',
    unsafe_allow_html=True,
)

summary_columns = st.columns(2, gap="small")

worst_case_figure = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Feature Noise", "Parameter Noise", "Runtime Noise"],
    shared_yaxes=True,
    horizontal_spacing=0.04,
)

for col_index, noise_name in enumerate(noise_columns, start=1):
    worst_case_table = (
        sweep_results.groupby(["selector", noise_name], as_index=False)[curve_metric].max()
    )
    for selector_name in selector_order:
        selector_slice = worst_case_table[worst_case_table["selector"] == selector_name]
        worst_case_figure.add_trace(
            go.Scatter(
                x=selector_slice[noise_name],
                y=selector_slice[curve_metric],
                mode="lines+markers",
                name=selector_name,
                legendgroup=f"worst-{selector_name}",
                showlegend=col_index == 1,
                line=dict(width=3, color=selector_colors[selector_name]),
                marker=dict(size=9, color=selector_colors[selector_name]),
            ),
            row=1,
            col=col_index,
        )
    worst_case_figure.update_xaxes(title_text=noise_name.replace("_", " "), row=1, col=col_index)

worst_case_figure.update_yaxes(
    title_text=f"worst-case {curve_metric.replace('_', ' ')}",
    row=1,
    col=1,
)
worst_case_figure.update_layout(
    height=420,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=36, b=10),
    font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    legend_title_text="",
)

with summary_columns[0]:
    st.plotly_chart(worst_case_figure, use_container_width=True)

sweep_rankings = (
    sweep_results.sort_values(
        ["feature_noise", "parameter_noise", "runtime_noise", "seed", curve_metric],
        ascending=[True, True, True, True, curve_metric != "accuracy"],
    ).copy()
)
sweep_rankings["rank"] = (
    sweep_rankings.groupby(["feature_noise", "parameter_noise", "runtime_noise", "seed"]).cumcount()
    + 1
)
sweep_rank_summary = (
    sweep_rankings.groupby("selector", as_index=False)
    .agg(
        win_rate=("rank", lambda values: float((values == 1).mean())),
        mean_rank=("rank", "mean"),
    )
    .sort_values(["win_rate", "mean_rank"], ascending=[False, True])
)

with summary_columns[1]:
    win_rate_chart = px.bar(
        sweep_rank_summary,
        x="selector",
        y="win_rate",
        color="selector",
        color_discrete_map=selector_colors,
        text=sweep_rank_summary["win_rate"].map(lambda value: f"{100 * value:.1f}%"),
        labels={"selector": "", "win_rate": "Win rate across full sweep"},
    )
    win_rate_chart.update_traces(textposition="outside")
    win_rate_chart.update_layout(
        height=420,
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        margin=dict(l=10, r=10, t=36, b=10),
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    )
    st.plotly_chart(win_rate_chart, use_container_width=True)

st.markdown(
    '<div class="section-title">Low vs High Messiness</div>',
    unsafe_allow_html=True,
)
messiness_table = sweep_results.copy()
messiness_table["noise_total"] = (
    messiness_table["feature_noise"]
    + messiness_table["parameter_noise"]
    + messiness_table["runtime_noise"]
)
low_cutoff = messiness_table["noise_total"].quantile(0.33)
high_cutoff = messiness_table["noise_total"].quantile(0.67)
messiness_table["messiness"] = "Mid"
messiness_table.loc[messiness_table["noise_total"] <= low_cutoff, "messiness"] = "Low"
messiness_table.loc[messiness_table["noise_total"] >= high_cutoff, "messiness"] = "High"
messiness_compare = (
    messiness_table[messiness_table["messiness"] != "Mid"]
    .groupby(["selector", "messiness"], as_index=False)[curve_metric]
    .mean()
)
messiness_chart = px.bar(
    messiness_compare,
    x="selector",
    y=curve_metric,
    color="messiness",
    barmode="group",
    color_discrete_map={"Low": "#0f6e76", "High": "#b44c32"},
    labels={"selector": "", curve_metric: curve_metric.replace("_", " "), "messiness": ""},
)
messiness_chart.update_layout(
    height=340,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
    font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
)
st.plotly_chart(messiness_chart, use_container_width=True)

st.markdown(
    '<div class="section-title">Features vs True Best Parameters</div>',
    unsafe_allow_html=True,
)
simulation_controls = st.columns((1.0, 1.0, 1.0, 1.0, 1.1, 1.1), gap="small")
with simulation_controls[0]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    seed = st.slider("Seed", min_value=0, max_value=100, value=7)
    st.markdown("</div>", unsafe_allow_html=True)
with simulation_controls[1]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    n_instances = st.slider("Instances", min_value=90, max_value=1500, value=360, step=30)
    st.markdown("</div>", unsafe_allow_html=True)
with simulation_controls[2]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    feature_noise = st.slider(
        "Feature noise",
        min_value=0.05,
        max_value=1.2,
        value=0.40,
        step=0.05,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with simulation_controls[3]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    parameter_noise = st.slider(
        "Parameter noise",
        min_value=0.01,
        max_value=0.25,
        value=0.08,
        step=0.01,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with simulation_controls[4]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    runtime_noise = st.slider(
        "Runtime noise",
        min_value=0.0,
        max_value=0.20,
        value=0.03,
        step=0.01,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with simulation_controls[5]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    selected_selector = st.selectbox(
        "Selected algorithm",
        options=[
            "DGCAC-inspired",
            "Cluster ISAC",
            "Classifier",
            "MLP Selector",
            "Regressor",
            "Oracle",
        ],
        index=0,
    )
    st.markdown("</div>", unsafe_allow_html=True)

selector_table, _, instance_table, portfolio_table = build_dashboard_data(
    n_instances=n_instances,
    feature_noise=feature_noise,
    parameter_noise=parameter_noise,
    runtime_noise=runtime_noise,
    seed=seed,
)

selector_key = selected_selector.lower().replace(" ", "_")
choice_column = f"choice_name_{selector_key}"
regret_column = f"regret_{selector_key}"

selected_param_table = portfolio_table.rename(
    columns={
        "config_name": choice_column,
        "param_1": "selected_param_1",
        "param_2": "selected_param_2",
    }
)[[choice_column, "selected_param_1", "selected_param_2"]]
instance_table = instance_table.merge(selected_param_table, on=choice_column, how="left")

true_left, true_right = st.columns(2, gap="small")
with true_left:
    st.plotly_chart(
        plot_parameter_surface(
            instance_table,
            color_column="ideal_param_1",
            title="True parameter 1",
        ),
        use_container_width=True,
    )
with true_right:
    st.plotly_chart(
        plot_parameter_surface(
            instance_table,
            color_column="ideal_param_2",
            title="True parameter 2",
        ),
        use_container_width=True,
    )

st.markdown(
    '<div class="section-title">Features vs Selected Best Parameters</div>',
    unsafe_allow_html=True,
)
selected_left, selected_right = st.columns(2, gap="small")
with selected_left:
    st.plotly_chart(
        plot_parameter_surface(
            instance_table,
            color_column="selected_param_1",
            title=f"{selected_selector} parameter 1",
            selector_name=selected_selector,
        ),
        use_container_width=True,
    )
with selected_right:
    st.plotly_chart(
        plot_parameter_surface(
            instance_table,
            color_column="selected_param_2",
            title=f"{selected_selector} parameter 2",
            selector_name=selected_selector,
        ),
        use_container_width=True,
    )
