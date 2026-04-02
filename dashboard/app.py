"""Streamlit dashboard for the portfolio benchmark."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from isac.analysis import evaluate_selectors, make_portfolio_table
from isac.core import PortfolioBenchmark
from isac.experiments import run_noise_sweep

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
    initial_sidebar_state="expanded",
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
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .hero-card, .metric-card {
        background: rgba(255, 252, 246, 0.85);
        border: 1px solid rgba(31, 42, 46, 0.08);
        border-radius: 22px;
        box-shadow: 0 18px 50px rgba(60, 53, 35, 0.10);
        backdrop-filter: blur(6px);
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
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
    .metric-label {
        font-size: 0.82rem;
        color: #7c6e58;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #122126;
        margin-top: 0.2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #122126;
        margin-bottom: 0.35rem;
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
) -> tuple[object, object, object, object]:
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
def build_noise_sweep_data(
    feature_noises: tuple[float, ...],
    parameter_noises: tuple[float, ...],
    runtime_noises: tuple[float, ...],
    n_instances: int,
    seeds: tuple[int, ...],
) -> object:
    return run_noise_sweep(
        feature_noises=list(feature_noises),
        parameter_noises=list(parameter_noises),
        runtime_noises=list(runtime_noises),
        n_instances=n_instances,
        seeds=list(seeds),
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### Simulation Controls")
    seed = st.slider("Seed", min_value=0, max_value=100, value=7)
    n_instances = st.slider("Instances", min_value=90, max_value=1500, value=360, step=30)
    feature_noise = st.slider(
        "Feature noise",
        min_value=0.05,
        max_value=1.2,
        value=0.40,
        step=0.05,
    )
    parameter_noise = st.slider(
        "Parameter noise",
        min_value=0.01,
        max_value=0.25,
        value=0.08,
        step=0.01,
    )
    runtime_noise = st.slider("Runtime noise", min_value=0.0, max_value=0.20, value=0.03, step=0.01)
    highlight_selector = st.selectbox(
        "Highlight selector",
        options=[
            "DGCAC-inspired",
            "Cluster ISAC",
            "Classifier",
            "Regressor",
            "Global Best",
            "Random",
            "Oracle",
        ],
        index=0,
    )
    st.markdown("### Sweep Controls")
    sweep_instances = st.slider("Sweep instances", min_value=120, max_value=720, value=240, step=60)
    sweep_seed_count = st.slider("Sweep seeds", min_value=2, max_value=8, value=4)


selector_table, action_table, instance_table, portfolio_table = build_dashboard_data(
    n_instances=n_instances,
    feature_noise=feature_noise,
    parameter_noise=parameter_noise,
    runtime_noise=runtime_noise,
    seed=seed,
)

best_non_oracle = selector_table.loc[selector_table["selector"] != "Oracle"].sort_values(
    by="avg_regret"
).iloc[0]
highlight_key = highlight_selector.lower().replace(" ", "_")
highlight_choice_column = f"choice_name_{highlight_key}"
highlight_regret_column = f"regret_{highlight_key}"

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Instance-Specific Portfolio Selection</div>
        <div class="hero-title">A live view of the simulation geometry</div>
        <p class="hero-copy">
            This dashboard shows how a fixed portfolio of parameter settings behaves
            across latent data regimes. Each point is a sampled instance; selectors
            must route from observable features alone.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Instances", f"{n_instances}")
with metric_cols[1]:
    metric_card("Best Heuristic", str(best_non_oracle["selector"]))
with metric_cols[2]:
    metric_card("Heuristic Avg. Regret", f"{best_non_oracle['avg_regret']:.3f}")
with metric_cols[3]:
    metric_card("Heuristic Accuracy", f"{100 * best_non_oracle['accuracy']:.1f}%")

left_col, right_col = st.columns((1.6, 1.0), gap="large")

with left_col:
    st.markdown(
        '<div class="section-title">Feature Space and Selector Decisions</div>',
        unsafe_allow_html=True,
    )
    scatter = px.scatter(
        instance_table,
        x="x",
        y="y",
        color="regime_label",
        symbol=highlight_choice_column,
        hover_data={
            "instance_id": True,
            "regime_id": True,
            "regime_label": False,
            "best_config": True,
            highlight_choice_column: True,
            highlight_regret_column: ":.3f",
            "x": False,
            "y": False,
        },
        category_orders={"regime_label": sorted(instance_table["regime_label"].unique())},
        color_discrete_map=REGIME_COLOR_MAP,
        labels={
            "x": "Embedding Axis 1",
            "y": "Embedding Axis 2",
            "regime_label": "Regime",
            highlight_choice_column: "Chosen config",
        },
    )
    scatter.update_traces(marker=dict(size=10, opacity=0.82, line=dict(width=0.4, color="#f7f3ea")))
    scatter.update_layout(
        height=520,
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        legend_title_text="",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown('<div class="section-title">Per-Selector Performance</div>', unsafe_allow_html=True)
    selector_bar = go.Figure()
    selector_bar.add_bar(
        x=selector_table["selector"],
        y=selector_table["avg_regret"],
        marker_color=["#122126", "#c8772a", "#7f8f8c", "#0f6e76"],
        text=[f"{value:.3f}" for value in selector_table["avg_regret"]],
        textposition="outside",
    )
    selector_bar.update_layout(
        height=320,
        yaxis_title="Average regret",
        xaxis_title="",
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    )
    st.plotly_chart(selector_bar, use_container_width=True)

with right_col:
    st.markdown('<div class="section-title">Portfolio Geometry</div>', unsafe_allow_html=True)
    portfolio_scatter = px.scatter(
        portfolio_table,
        x="param_1",
        y="param_2",
        text="config_name",
        color="config_name",
        color_discrete_sequence=["#0f6e76", "#d8892b", "#b44c32", "#6e7f52"],
        labels={"param_1": "Parameter 1", "param_2": "Parameter 2"},
    )
    portfolio_scatter.update_traces(marker=dict(size=24, line=dict(width=1.5, color="#fffaf0")))
    portfolio_scatter.update_traces(textposition="top center")
    portfolio_scatter.update_layout(
        height=250,
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_range=[0.0, 1.0],
        yaxis_range=[0.0, 1.0],
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    )
    st.plotly_chart(portfolio_scatter, use_container_width=True)

    st.markdown('<div class="section-title">Action Mix</div>', unsafe_allow_html=True)
    action_view = action_table[action_table["selector"] == highlight_selector]
    action_bar = px.bar(
        action_view,
        x="config_name",
        y="count",
        color="config_name",
        color_discrete_sequence=["#0f6e76", "#d8892b", "#b44c32", "#6e7f52"],
        labels={"count": "Selections", "config_name": "Config"},
    )
    action_bar.update_layout(
        height=250,
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
    )
    st.plotly_chart(action_bar, use_container_width=True)

    st.markdown('<div class="section-title">Selector Scorecard</div>', unsafe_allow_html=True)
    scorecard = selector_table.copy()
    scorecard["accuracy"] = (100 * scorecard["accuracy"]).map(lambda value: f"{value:.1f}%")
    scorecard["avg_runtime"] = scorecard["avg_runtime"].map(lambda value: f"{value:.3f}")
    scorecard["avg_regret"] = scorecard["avg_regret"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        scorecard,
        use_container_width=True,
        hide_index=True,
        column_config={
            "selector": "Selector",
            "avg_runtime": "Avg. runtime",
            "avg_regret": "Avg. regret",
            "accuracy": "Accuracy",
        },
    )

st.markdown('<div class="section-title">Instance-Level Results</div>', unsafe_allow_html=True)
display_columns = [
    "instance_id",
    "regime_id",
    "best_config",
    highlight_choice_column,
    highlight_regret_column,
    "ideal_param_1",
    "ideal_param_2",
    "feature_1",
    "feature_2",
]
renamed = instance_table[display_columns].rename(
    columns={
        highlight_choice_column: "chosen_config",
        highlight_regret_column: "selector_regret",
    }
)
st.dataframe(renamed, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">3D Noise Sweep</div>', unsafe_allow_html=True)
sweep_results = build_noise_sweep_data(
    feature_noises=(0.10, 0.40, 0.80),
    parameter_noises=(0.02, 0.08, 0.16),
    runtime_noises=(0.00, 0.03, 0.08),
    n_instances=sweep_instances,
    seeds=tuple(range(sweep_seed_count)),
)

sweep_selector = st.selectbox(
    "Sweep selector",
    options=[name for name in selector_table["selector"] if name != "Oracle"],
    index=0,
)
sweep_runtime = st.selectbox(
    "Fix runtime noise",
    options=sorted(sweep_results["runtime_noise"].unique().tolist()),
    index=1,
)

sweep_slice = sweep_results[
    (sweep_results["selector"] == sweep_selector)
    & (sweep_results["runtime_noise"] == sweep_runtime)
].copy()
sweep_agg = (
    sweep_slice.groupby(["feature_noise", "parameter_noise"], as_index=False)["avg_regret"]
    .mean()
    .sort_values(["feature_noise", "parameter_noise"])
)
heatmap = px.density_heatmap(
    sweep_agg,
    x="feature_noise",
    y="parameter_noise",
    z="avg_regret",
    histfunc="avg",
    text_auto=".3f",
    color_continuous_scale=["#f8f2df", "#e0b35b", "#b44c32", "#5e1f14"],
    labels={
        "feature_noise": "Feature noise",
        "parameter_noise": "Parameter noise",
        "avg_regret": "Avg. regret",
    },
)
heatmap.update_layout(
    height=360,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
    font=dict(family="Avenir Next, Helvetica Neue, sans-serif", color="#1f2a2e"),
)
st.plotly_chart(heatmap, use_container_width=True)

sweep_summary = (
    sweep_results[sweep_results["selector"] != "Oracle"]
    .groupby("selector", as_index=False)
    .agg(
        mean_avg_regret=("avg_regret", "mean"),
        worst_avg_regret=("avg_regret", "max"),
        mean_accuracy=("accuracy", "mean"),
    )
    .sort_values("mean_avg_regret")
)
sweep_summary["mean_accuracy"] = sweep_summary["mean_accuracy"].map(
    lambda value: f"{100 * value:.1f}%"
)
sweep_summary["mean_avg_regret"] = sweep_summary["mean_avg_regret"].map(
    lambda value: f"{value:.3f}"
)
sweep_summary["worst_avg_regret"] = sweep_summary["worst_avg_regret"].map(
    lambda value: f"{value:.3f}"
)
st.dataframe(sweep_summary, use_container_width=True, hide_index=True)
