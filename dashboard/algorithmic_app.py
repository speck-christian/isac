"""Streamlit dashboard for the concrete algorithmic benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PRECOMPUTED_ALGORITHMIC_SUMMARY_PATH = Path("artifacts/algorithmic/dashboard_selector_summary.csv")
PRECOMPUTED_ALGORITHMIC_TRACE_PATH = Path("artifacts/algorithmic/dashboard_trace_table.csv")
PRECOMPUTED_ALGORITHMIC_METADATA_PATH = Path("artifacts/algorithmic/dashboard_metadata.json")
PRECOMPUTED_ALGORITHMIC_ROBUSTNESS_PATH = Path("artifacts/algorithmic/robustness_sweep.csv")
PRECOMPUTED_ALGORITHMIC_ROBUSTNESS_METADATA_PATH = Path(
    "artifacts/algorithmic/robustness_metadata.json"
)

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
    page_title="ISAC Algorithmic Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(123, 143, 61, 0.16), transparent 32%),
            radial-gradient(circle at top right, rgba(15, 110, 118, 0.18), transparent 28%),
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
def load_algorithmic_dashboard_data(
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    selector_table = pd.read_csv(PRECOMPUTED_ALGORITHMIC_SUMMARY_PATH)
    trace_table = pd.read_csv(PRECOMPUTED_ALGORITHMIC_TRACE_PATH)
    metadata = json.loads(PRECOMPUTED_ALGORITHMIC_METADATA_PATH.read_text())
    return selector_table, trace_table, metadata


@st.cache_data(show_spinner=False)
def load_algorithmic_robustness_data(
) -> tuple[pd.DataFrame | None, dict[str, float | int | str] | None]:
    if not PRECOMPUTED_ALGORITHMIC_ROBUSTNESS_PATH.exists():
        return None, None
    robustness_table = pd.read_csv(PRECOMPUTED_ALGORITHMIC_ROBUSTNESS_PATH)
    robustness_metadata = json.loads(PRECOMPUTED_ALGORITHMIC_ROBUSTNESS_METADATA_PATH.read_text())
    return robustness_table, robustness_metadata

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Concrete Algorithm Configuration</div>
        <div class="hero-title">Algorithmic benchmark dashboard</div>
        <p class="hero-copy">
            This dashboard evaluates selectors on a concrete adaptive forecasting
            algorithm with tunable parameters <code>alpha</code>, <code>beta</code>,
            and <code>gate</code>. Each selector must first derive a portfolio
            offline, then infer which parameter setting to deploy online from
            changing feature data. When SMAC3 is installed, the dashboard can
            use true offline parameter search instead of only lightweight random
            search.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

selector_table, trace_table, algorithmic_metadata = load_algorithmic_dashboard_data()
selector_table = selector_table[selector_table["selector"].isin(SELECTOR_ORDER)].copy()
trace_table = trace_table[trace_table["selector"].isin(SELECTOR_ORDER)].copy()
robustness_table, robustness_metadata = load_algorithmic_robustness_data()

metadata_cols = st.columns(4, gap="small")
metadata_cards = [
    ("Artifact tag", str(algorithmic_metadata.get("artifact_tag", "default"))),
    ("Portfolio search", str(algorithmic_metadata.get("portfolio_source", "n/a"))),
    (
        "Seed / episodes",
        f"{algorithmic_metadata.get('seed', 'n/a')} / "
        f"{algorithmic_metadata.get('n_episodes', 'n/a')}",
    ),
    (
        "Horizon / trials",
        f"{algorithmic_metadata.get('horizon', 'n/a')} / "
        f"{algorithmic_metadata.get('portfolio_trials', 'n/a')}",
    ),
]
for column, (label, value) in zip(metadata_cols, metadata_cards, strict=True):
    with column:
        st.markdown(
            f"""
            <div class="control-card">
                <div style="font-weight: 700; color: #122126; margin-bottom: 0.15rem;">{label}</div>
                <div style="color: #415157; font-size: 0.98rem;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption(
    "This dashboard only reads stored artifacts. Refresh them offline with "
    "`scripts/run_algorithmic_dashboard_data.py`, then use this page to inspect the "
    "saved selector rankings and traces."
)
st.caption(
    "Portfolio size is now selected offline from validation data rather than fixed "
    "at a shared maximum, so the charts below show the derived size each selector kept."
)

st.markdown('<div class="section-title">Selector Performance</div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="small")

performance_chart = px.bar(
    selector_table.sort_values("avg_regret"),
    x="selector",
    y="avg_regret",
    color="selector",
    color_discrete_map=SELECTOR_COLORS,
    labels={"selector": "", "avg_regret": "Average regret"},
)
performance_chart.update_layout(
    height=360,
    showlegend=False,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
)
with left:
    st.plotly_chart(performance_chart, use_container_width=True)

portfolio_chart = px.bar(
    selector_table.sort_values("portfolio_size", ascending=False),
    x="selector",
    y="portfolio_size",
    color="selector",
    color_discrete_map=SELECTOR_COLORS,
    labels={"selector": "", "portfolio_size": "Derived portfolio size"},
)
portfolio_chart.update_layout(
    height=360,
    showlegend=False,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
)
with right:
    st.plotly_chart(portfolio_chart, use_container_width=True)

st.caption(
    "Single-run artifact summary. In the current saved run, the strongest selectors "
    "settled around portfolio sizes 7-8, while `DGCAC-inspired` selected a much smaller "
    "portfolio of 2."
)

summary_table = selector_table.sort_values("avg_regret").copy()
summary_table["avg_loss"] = summary_table["avg_loss"].map(lambda value: f"{value:.6f}")
summary_table["avg_regret"] = summary_table["avg_regret"].map(lambda value: f"{value:.6f}")
summary_table["portfolio_size"] = summary_table["portfolio_size"].astype(int)
st.dataframe(
    summary_table[["selector", "portfolio_source", "portfolio_size", "avg_loss", "avg_regret"]],
    use_container_width=True,
    hide_index=True,
)

if robustness_table is not None and robustness_metadata is not None:
    st.markdown('<div class="section-title">Robustness Across Seeds</div>', unsafe_allow_html=True)
    st.caption(
        "Precomputed multi-seed algorithmic sweep. Refresh offline with "
        "`scripts/run_algorithmic_robustness_sweep.py`."
    )
    st.caption(
        "Robustness settings: "
        f"seeds={robustness_metadata.get('seeds', [])}, "
        f"episodes={robustness_metadata.get('n_episodes', 'n/a')}, "
        f"horizon={robustness_metadata.get('horizon', 'n/a')}, "
        f"portfolio_source={robustness_metadata.get('portfolio_source', 'n/a')}, "
        f"trials={robustness_metadata.get('portfolio_trials', 'n/a')}."
    )
    robustness_table = robustness_table[
        robustness_table["selector"].isin(SELECTOR_ORDER)
    ].copy()
    robustness_summary = (
        robustness_table.groupby("selector", as_index=False)
        .agg(
            avg_regret_mean=("avg_regret", "mean"),
            avg_regret_std=("avg_regret", "std"),
            avg_loss_mean=("avg_loss", "mean"),
            avg_loss_std=("avg_loss", "std"),
            portfolio_size_mean=("portfolio_size", "mean"),
            portfolio_size_min=("portfolio_size", "min"),
            portfolio_size_max=("portfolio_size", "max"),
        )
    )
    robustness_chart = px.bar(
        robustness_summary.sort_values("avg_regret_mean"),
        x="selector",
        y="avg_regret_mean",
        color="selector",
        color_discrete_map=SELECTOR_COLORS,
        error_y="avg_regret_std",
        labels={"selector": "", "avg_regret_mean": "Mean avg regret across seeds"},
    )
    robustness_chart.update_layout(
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,252,246,0.55)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(robustness_chart, use_container_width=True)

    robustness_view = robustness_summary.sort_values("avg_regret_mean").copy()
    robustness_view["avg_regret_mean"] = robustness_view["avg_regret_mean"].map(
        lambda value: f"{value:.6f}"
    )
    robustness_view["avg_regret_std"] = robustness_view["avg_regret_std"].fillna(0.0).map(
        lambda value: f"{value:.6f}"
    )
    robustness_view["avg_loss_mean"] = robustness_view["avg_loss_mean"].map(
        lambda value: f"{value:.6f}"
    )
    robustness_view["avg_loss_std"] = robustness_view["avg_loss_std"].fillna(0.0).map(
        lambda value: f"{value:.6f}"
    )
    robustness_view["portfolio_size_mean"] = robustness_view["portfolio_size_mean"].map(
        lambda value: f"{value:.2f}"
    )
    robustness_view["portfolio_size_min"] = robustness_view["portfolio_size_min"].astype(int)
    robustness_view["portfolio_size_max"] = robustness_view["portfolio_size_max"].astype(int)
    st.dataframe(
        robustness_view[
            [
                "selector",
                "avg_regret_mean",
                "avg_regret_std",
                "avg_loss_mean",
                "avg_loss_std",
                "portfolio_size_mean",
                "portfolio_size_min",
                "portfolio_size_max",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown('<div class="section-title">Action Traces</div>', unsafe_allow_html=True)
selector_name = st.selectbox("Trace selector", options=SELECTOR_ORDER, index=0)
episode_id = st.slider(
    "Episode",
    min_value=0,
    max_value=max(int(trace_table["episode_id"].max()), 0),
    value=0,
)
selector_trace = trace_table[
    (trace_table["selector"] == selector_name) & (trace_table["episode_id"] == episode_id)
].copy()

trace_left, trace_right = st.columns(2, gap="small")

loss_chart = px.line(
    selector_trace,
    x="timestep",
    y=["loss", "best_loss"],
    labels={"value": "Loss", "variable": "", "timestep": "Timestep"},
    color_discrete_map={"loss": "#b44c32", "best_loss": "#0f6e76"},
)
loss_chart.update_layout(
    height=360,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
)
with trace_left:
    st.plotly_chart(loss_chart, use_container_width=True)

param_chart = px.line(
    selector_trace,
    x="timestep",
    y=["selected_param_1", "selected_param_2", "selected_param_3"],
    labels={"value": "Parameter value", "variable": "", "timestep": "Timestep"},
    color_discrete_map={
        "selected_param_1": "#0f6e76",
        "selected_param_2": "#d8892b",
        "selected_param_3": "#7b8f3d",
    },
)
param_chart.update_layout(
    height=360,
    paper_bgcolor="rgba(255,255,255,0.0)",
    plot_bgcolor="rgba(255,252,246,0.55)",
    margin=dict(l=10, r=10, t=10, b=10),
)
with trace_right:
    st.plotly_chart(param_chart, use_container_width=True)

st.markdown('<div class="section-title">Selector Table</div>', unsafe_allow_html=True)
st.dataframe(
    selector_table.sort_values("avg_regret").reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)
