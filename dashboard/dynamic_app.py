"""Streamlit dashboard for dynamic ISAC scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from isac.analysis import evaluate_dynamic_selectors
from isac.core import DynamicPortfolioBenchmark

PRECOMPUTED_DYNAMIC_SWEEP_PATH = Path("artifacts/dynamic/seed_episode_sweep.csv")
DEFAULT_DYNAMIC_SWEEP = {
    "horizon": 16,
    "drift_scale": 0.22,
    "regime_switch_prob": 0.18,
    "switching_cost": 0.04,
    "observation_noise": 0.10,
    "missing_feature_prob": 0.22,
    "multimodal_surface_scale": 0.30,
}

SELECTOR_ORDER = [
    "Temporal Soft Cluster ISAC",
    "DGCAC-inspired",
    "Cluster ISAC",
    "Privileged Classifier",
    "MLP Selector",
    "Regressor",
]
SELECTOR_COLORS = {
    "Temporal Soft Cluster ISAC": "#7b8f3d",
    "DGCAC-inspired": "#0f6e76",
    "Cluster ISAC": "#d8892b",
    "Privileged Classifier": "#b44c32",
    "MLP Selector": "#8c5e3c",
    "Regressor": "#6e7f52",
}

st.set_page_config(
    page_title="ISAC Dynamic Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(180, 76, 50, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(15, 110, 118, 0.20), transparent 26%),
            linear-gradient(180deg, #f6f2e8 0%, #efe8da 100%);
        color: #1f2a2e;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 0.6rem;
        max-width: 1440px;
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
        max-width: 62rem;
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
def build_dynamic_dashboard_data(
    *,
    n_episodes: int,
    horizon: int,
    drift_scale: float,
    regime_switch_prob: float,
    switching_cost: float,
    observation_noise: float,
    missing_feature_prob: float,
    multimodal_surface_scale: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = DynamicPortfolioBenchmark(
        horizon=horizon,
        drift_scale=drift_scale,
        regime_switch_prob=regime_switch_prob,
        switching_cost=switching_cost,
        observation_noise=observation_noise,
        missing_feature_prob=missing_feature_prob,
        multimodal_surface_scale=multimodal_surface_scale,
        seed=seed,
    )
    selector_table, trace_table = evaluate_dynamic_selectors(
        benchmark=benchmark,
        n_episodes=n_episodes,
        seed=seed,
    )
    return selector_table, trace_table


@st.cache_data(show_spinner=False)
def load_dynamic_seed_episode_sweep() -> pd.DataFrame:
    return pd.read_csv(PRECOMPUTED_DYNAMIC_SWEEP_PATH)


st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Dynamic Instance-Specific Configuration</div>
        <div class="hero-title">Evolving instance dashboard</div>
        <p class="hero-copy">
            This dashboard focuses on scenarios where features drift, the true best
            parameter choice changes over time, switching configurations carries a cost,
            and performance surfaces can become multimodal under partial observability.
            The Privileged Classifier is included as a truth-aware comparator against the
            unlabeled alternatives.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

controls = st.columns((0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0), gap="small")
with controls[0]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    seed = st.slider("Seed", min_value=0, max_value=100, value=11)
    st.markdown("</div>", unsafe_allow_html=True)
with controls[1]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    n_episodes = st.slider("Episodes", min_value=20, max_value=200, value=80, step=20)
    st.markdown("</div>", unsafe_allow_html=True)
with controls[2]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    horizon = st.slider("Horizon", min_value=8, max_value=40, value=16, step=4)
    st.markdown("</div>", unsafe_allow_html=True)
with controls[3]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    drift_scale = st.slider("Drift scale", min_value=0.05, max_value=0.40, value=0.22, step=0.01)
    st.markdown("</div>", unsafe_allow_html=True)
with controls[4]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    regime_switch_prob = st.slider(
        "Regime switch prob",
        min_value=0.0,
        max_value=0.50,
        value=0.18,
        step=0.01,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with controls[5]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    switching_cost = st.slider(
        "Switching cost",
        min_value=0.0,
        max_value=0.20,
        value=0.04,
        step=0.01,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with controls[6]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    observation_noise = st.slider(
        "Observation noise",
        min_value=0.0,
        max_value=0.40,
        value=0.10,
        step=0.01,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with controls[7]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    missing_feature_prob = st.slider(
        "Missing feature prob",
        min_value=0.0,
        max_value=0.80,
        value=0.22,
        step=0.02,
    )
    st.markdown("</div>", unsafe_allow_html=True)
controls_bottom = st.columns((1.2, 1.0), gap="small")
with controls_bottom[0]:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    multimodal_surface_scale = st.slider(
        "Multimodal surface scale",
        min_value=0.0,
        max_value=0.80,
        value=0.30,
        step=0.05,
    )
    st.markdown("</div>", unsafe_allow_html=True)
with controls_bottom[1]:
    st.markdown(
        """
        <div class="control-card">
            <div
                style="font-weight: 700; color: #122126; margin-bottom: 0.2rem;"
            >
                Fixed defaults
            </div>
            <div style="color: #415157; font-size: 0.95rem; line-height: 1.35;">
                Dynamic runs keep latent feature noise, parameter noise, and runtime noise
                at their benchmark defaults so the controls stay focused on drift, sensing,
                switching, and multimodal performance structure.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

selector_table, trace_table = build_dynamic_dashboard_data(
    n_episodes=n_episodes,
    horizon=horizon,
    drift_scale=drift_scale,
    regime_switch_prob=regime_switch_prob,
    switching_cost=switching_cost,
    observation_noise=observation_noise,
    missing_feature_prob=missing_feature_prob,
    multimodal_surface_scale=multimodal_surface_scale,
    seed=seed,
)
selector_table = selector_table[selector_table["selector"].isin(SELECTOR_ORDER)].copy()
trace_table = trace_table[trace_table["selector"].isin(SELECTOR_ORDER)].copy()

seed_episode_sweep = load_dynamic_seed_episode_sweep()

st.markdown(
    '<div class="section-title">Aggregate Dynamic Performance</div>',
    unsafe_allow_html=True,
)
summary_left, summary_right = st.columns(2, gap="small")

penalty_chart = px.bar(
    selector_table.sort_values("avg_total_penalty"),
    x="selector",
    y="avg_total_penalty",
    color="selector",
    color_discrete_map=SELECTOR_COLORS,
    labels={"selector": "", "avg_total_penalty": "Average total penalty"},
)
penalty_chart.update_layout(
    height=360,
    showlegend=False,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
)
with summary_left:
    st.plotly_chart(penalty_chart, use_container_width=True)

decomposition_table = selector_table.melt(
    id_vars=["selector"],
    value_vars=["avg_regret", "avg_switch_cost"],
    var_name="component",
    value_name="value",
)
decomposition_chart = px.bar(
    decomposition_table,
    x="selector",
    y="value",
    color="component",
    barmode="stack",
    color_discrete_map={
        "avg_regret": "#0f6e76",
        "avg_switch_cost": "#b44c32",
    },
    labels={"selector": "", "value": "Penalty components", "component": ""},
)
decomposition_chart.update_layout(
    height=360,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
)
with summary_right:
    st.plotly_chart(decomposition_chart, use_container_width=True)

st.markdown(
    '<div class="section-title">Performance Across Seeds and Episodes</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Precomputed dynamic robustness sweep over seeds 0-7 and episode counts 20-200 "
    "under the default dynamic scenario."
)
st.caption(
    "Fixed sweep settings: "
    f"horizon={DEFAULT_DYNAMIC_SWEEP['horizon']}, "
    f"drift={DEFAULT_DYNAMIC_SWEEP['drift_scale']:.2f}, "
    f"switch_prob={DEFAULT_DYNAMIC_SWEEP['regime_switch_prob']:.2f}, "
    f"switch_cost={DEFAULT_DYNAMIC_SWEEP['switching_cost']:.2f}, "
    f"obs_noise={DEFAULT_DYNAMIC_SWEEP['observation_noise']:.2f}, "
    f"missing={DEFAULT_DYNAMIC_SWEEP['missing_feature_prob']:.2f}, "
    f"multimodal={DEFAULT_DYNAMIC_SWEEP['multimodal_surface_scale']:.2f}."
)
sweep_summary = (
    seed_episode_sweep.groupby(["selector", "episodes"], as_index=False)
    .agg(
        avg_total_penalty_mean=("avg_total_penalty", "mean"),
        avg_total_penalty_std=("avg_total_penalty", "std"),
        avg_regret_mean=("avg_regret", "mean"),
        avg_regret_std=("avg_regret", "std"),
        avg_switch_cost_mean=("avg_switch_cost", "mean"),
        avg_switch_cost_std=("avg_switch_cost", "std"),
    )
)
seed_episode_figure = go.Figure()
for selector_name in SELECTOR_ORDER:
    selector_slice = sweep_summary[
        sweep_summary["selector"] == selector_name
    ].sort_values("episodes")
    seed_episode_figure.add_trace(
        go.Scatter(
            x=selector_slice["episodes"],
            y=selector_slice["avg_total_penalty_mean"],
            mode="lines+markers",
            name=selector_name,
            line=dict(width=3, color=SELECTOR_COLORS[selector_name]),
            marker=dict(size=9, color=SELECTOR_COLORS[selector_name]),
            error_y=dict(
                type="data",
                array=selector_slice["avg_total_penalty_std"],
                thickness=1.2,
                width=4,
                color=SELECTOR_COLORS[selector_name],
            ),
            customdata=selector_slice[
                [
                    "avg_regret_mean",
                    "avg_switch_cost_mean",
                    "avg_total_penalty_std",
                ]
            ].to_numpy(),
            hovertemplate=(
                "episodes=%{x}<br>"
                "avg_total_penalty=%{y:.3f}<br>"
                "avg_regret=%{customdata[0]:.3f}<br>"
                "avg_switch_cost=%{customdata[1]:.3f}<br>"
                "seed_std=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
seed_episode_figure.update_layout(
    height=420,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Evaluation episodes",
    yaxis_title="Average total penalty",
    legend_title_text="",
)
st.plotly_chart(seed_episode_figure, use_container_width=True)

st.markdown(
    '<div class="section-title">How Performance Changes Over Time</div>',
    unsafe_allow_html=True,
)
timestep_table = (
    trace_table.groupby(["selector", "timestep"], as_index=False)[["regret", "switch_cost"]].mean()
)
timestep_figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Mean regret by timestep", "Mean switch cost by timestep"],
    shared_xaxes=False,
    horizontal_spacing=0.08,
)
for col_index, metric in enumerate(["regret", "switch_cost"], start=1):
    for selector_name in SELECTOR_ORDER:
        selector_slice = timestep_table[timestep_table["selector"] == selector_name]
        timestep_figure.add_trace(
            go.Scatter(
                x=selector_slice["timestep"],
                y=selector_slice[metric],
                mode="lines+markers",
                name=selector_name,
                legendgroup=selector_name,
                showlegend=col_index == 1,
                line=dict(width=3, color=SELECTOR_COLORS[selector_name]),
                marker=dict(size=8, color=SELECTOR_COLORS[selector_name]),
            ),
            row=1,
            col=col_index,
        )
timestep_figure.update_xaxes(title_text="Timestep", row=1, col=1)
timestep_figure.update_xaxes(title_text="Timestep", row=1, col=2)
timestep_figure.update_yaxes(title_text="Mean regret", row=1, col=1)
timestep_figure.update_yaxes(title_text="Mean switch cost", row=1, col=2)
timestep_figure.update_layout(
    height=420,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=36, b=10),
    legend_title_text="",
)
st.plotly_chart(timestep_figure, use_container_width=True)

st.markdown(
    '<div class="section-title">Episode Timeline Explorer</div>',
    unsafe_allow_html=True,
)
timeline_controls = st.columns((0.9, 1.4), gap="small")
with timeline_controls[0]:
    episode_id = st.slider(
        "Episode",
        min_value=0,
        max_value=max(int(trace_table["episode_id"].max()), 0),
        value=0,
    )
with timeline_controls[1]:
    selected_timeline_selectors = st.multiselect(
        "Selectors to overlay",
        options=SELECTOR_ORDER,
        default=["MLP Selector", "Regressor", "DGCAC-inspired"],
    )

episode_trace = trace_table[trace_table["episode_id"] == episode_id].copy()
truth_trace = (
    episode_trace[episode_trace["selector"] == SELECTOR_ORDER[0]][
        [
            "timestep",
            "regime_id",
            "best_config",
            "feature_1",
            "feature_2",
            "latent_feature_1",
            "latent_feature_2",
            "mask_1",
            "mask_2",
            "ideal_param_1",
            "ideal_param_2",
        ]
    ]
    .drop_duplicates(subset=["timestep"])
    .sort_values("timestep")
)

feature_figure = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=[
        "Observed vs latent features",
        "Observation missingness",
        "Ideal parameter drift",
    ],
    vertical_spacing=0.12,
)
feature_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["feature_1"],
        mode="lines+markers",
        name="observed_feature_1",
        line=dict(width=3, color="#0f6e76"),
    ),
    row=1,
    col=1,
)
feature_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["latent_feature_1"],
        mode="lines+markers",
        name="latent_feature_1",
        line=dict(width=3, color="#0f6e76", dash="dot"),
    ),
    row=1,
    col=1,
)
feature_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["feature_2"],
        mode="lines+markers",
        name="observed_feature_2",
        line=dict(width=3, color="#d8892b"),
    ),
    row=1,
    col=1,
)
feature_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["latent_feature_2"],
        mode="lines+markers",
        name="latent_feature_2",
        line=dict(width=3, color="#d8892b", dash="dot"),
    ),
    row=1,
    col=1,
)
feature_figure.add_trace(
    go.Bar(
        x=truth_trace["timestep"],
        y=1.0 - truth_trace["mask_1"],
        name="missing_feature_1",
        marker_color="#0f6e76",
    ),
    row=2,
    col=1,
)
feature_figure.add_trace(
    go.Bar(
        x=truth_trace["timestep"],
        y=1.0 - truth_trace["mask_2"],
        name="missing_feature_2",
        marker_color="#d8892b",
    ),
    row=2,
    col=1,
)
feature_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["ideal_param_1"],
        mode="lines+markers",
        name="ideal_param_1",
        line=dict(width=3, color="#b44c32"),
    ),
    row=3,
    col=1,
)
feature_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["ideal_param_2"],
        mode="lines+markers",
        name="ideal_param_2",
        line=dict(width=3, color="#6e7f52"),
    ),
    row=3,
    col=1,
)
feature_figure.update_xaxes(title_text="Timestep", row=3, col=1)
feature_figure.update_yaxes(title_text="Feature value", row=1, col=1)
feature_figure.update_yaxes(title_text="Missing flag", row=2, col=1)
feature_figure.update_yaxes(title_text="Ideal parameter", row=3, col=1)
feature_figure.update_layout(
    height=560,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=36, b=10),
)

timeline_figure = go.Figure()
timeline_figure.add_trace(
    go.Scatter(
        x=truth_trace["timestep"],
        y=truth_trace["best_config"],
        mode="lines+markers",
        name="True best config",
        line=dict(width=4, color="#122126", dash="dot"),
        marker=dict(size=9, color="#122126"),
    )
)
for selector_name in selected_timeline_selectors:
    selector_slice = episode_trace[
        episode_trace["selector"] == selector_name
    ].sort_values("timestep")
    timeline_figure.add_trace(
        go.Scatter(
            x=selector_slice["timestep"],
            y=selector_slice["action"],
            mode="lines+markers",
            name=selector_name,
            line=dict(width=3, color=SELECTOR_COLORS[selector_name]),
            marker=dict(size=9, color=SELECTOR_COLORS[selector_name]),
            customdata=selector_slice[["regret", "switch_cost", "total_penalty"]].to_numpy(),
            hovertemplate=(
                "t=%{x}<br>action=%{y}<br>"
                "regret=%{customdata[0]:.3f}<br>"
                "switch=%{customdata[1]:.3f}<br>"
                "total=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
timeline_figure.update_layout(
    height=420,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Timestep",
    yaxis_title="Chosen config index",
)

timeline_left, timeline_right = st.columns(2, gap="small")
with timeline_left:
    st.plotly_chart(feature_figure, use_container_width=True)
with timeline_right:
    st.plotly_chart(timeline_figure, use_container_width=True)
