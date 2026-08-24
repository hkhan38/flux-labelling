"""Streamlit labelling app for greenhouse gas flux QC.

Elisabeth reviews each chamber closure measurement (mmnt_id), inspects the
CO2 / N2O / CH4 fits and chamber conditions, then assigns QC reason codes
that will be used as training labels for a downstream ML model.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = "./raw/dt_flux_2024-07-07.csv"
OUTPUT_DIR = "./output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "flux_labels.csv")

REASON_CODES = [
    "PASS",
    "LOW_SIGNAL",
    "CO2_LIMITATION",
    "POOR_FIT",
    "TEMPERATURE_CHANGE",
    "PRESSURE_CHANGE",
    "POSSIBLE_LEAK",
    "UNEXPECTED_RESPONSE",
    "INSTRUMENT_WARNING",
]

CONTEXT_COLS = [
    "mmnt_id",
    "datect",
    "soil",
    "vegetation",
    "chamber",
    "light",
    "PPFD_IN_ch",
    "VWC",
]

st.set_page_config(page_title="Flux QC Labelling", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["mmnt_id", "t"])
    return df


def load_existing_labels():
    if os.path.exists(OUTPUT_PATH):
        labels_df = pd.read_csv(OUTPUT_PATH, dtype=str).fillna("")
        return dict(zip(labels_df["mmnt_id"], labels_df["reason_codes"]))
    return {}


def save_labels(labels: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df = pd.DataFrame(
        [{"mmnt_id": mid, "reason_codes": codes} for mid, codes in labels.items()]
    )
    out_df.to_csv(OUTPUT_PATH, index=False)


def line_trace(x, y, name, color, dash=None):
    line = dict(color=color)
    if dash:
        line["dash"] = dash
    return go.Scatter(x=x, y=y, mode="lines", name=name, line=line)


def init_state():
    if "df" not in st.session_state:
        st.session_state.df = load_data()
        st.session_state.mmnt_ids = sorted(st.session_state.df["mmnt_id"].unique())
    if "labels" not in st.session_state:
        st.session_state.labels = load_existing_labels()
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0


def go_to_next_unlabelled():
    mmnt_ids = st.session_state.mmnt_ids
    n = len(mmnt_ids)
    start = st.session_state.current_idx
    for offset in range(1, n + 1):
        idx = (start + offset) % n
        if mmnt_ids[idx] not in st.session_state.labels:
            st.session_state.current_idx = idx
            return
    st.toast("All measurements are labelled!")


init_state()

df = st.session_state.df
mmnt_ids = st.session_state.mmnt_ids
n_total = len(mmnt_ids)
labels = st.session_state.labels

# ---------------------------------------------------------------------------
# Sidebar: progress summary + reason code breakdown
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Labelling progress")
    n_labelled = len(labels)
    st.metric("Labelled", f"{n_labelled} / {n_total}")
    st.progress(n_labelled / n_total if n_total else 0)
    st.caption(f"{n_total - n_labelled} remaining")

    st.divider()
    st.subheader("Reason code counts")
    code_counts = {code: 0 for code in REASON_CODES}
    for codes_str in labels.values():
        if codes_str:
            for code in codes_str.split("|"):
                if code in code_counts:
                    code_counts[code] += 1
    for code, count in code_counts.items():
        st.write(f"**{code}**: {count}")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
current_idx = st.session_state.current_idx
current_mmnt_id = mmnt_ids[current_idx]

st.title("Flux QC Labelling")
st.subheader(f"Measurement {current_idx + 1} of {n_total}")
st.progress((current_idx + 1) / n_total)

nav_col1, nav_col2, nav_col3, _ = st.columns([1, 1, 2, 4])
with nav_col1:
    if st.button("Previous", disabled=(current_idx == 0), width='stretch'):
        st.session_state.current_idx = max(0, current_idx - 1)
        st.rerun()
with nav_col2:
    if st.button("Next", disabled=(current_idx == n_total - 1), width='stretch'):
        st.session_state.current_idx = min(n_total - 1, current_idx + 1)
        st.rerun()
with nav_col3:
    if st.button("Skip to next unlabelled", width='stretch'):
        go_to_next_unlabelled()
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Context bar
# ---------------------------------------------------------------------------
group = df[df["mmnt_id"] == current_mmnt_id]
first_row = group.iloc[0]
plot_df = group[group["exclude"] == False]  # noqa: E712

CONTEXT_COL_WIDTHS = {
    "mmnt_id": 2.2,
    "datect": 2,
    "soil": 1,
    "vegetation": 1,
    "chamber": 1,
    "light": 1,
    "PPFD_IN_ch": 1,
    "VWC": 1,
}
context_cols = st.columns([CONTEXT_COL_WIDTHS[c] for c in CONTEXT_COLS])
for col, field in zip(context_cols, CONTEXT_COLS):
    value = first_row[field]
    if isinstance(value, float):
        value = f"{value:.3g}"
    col.markdown(f"**{field}**")
    col.markdown(
        f"<div style='word-break:break-word; white-space:normal; font-size:0.95rem;'>{value}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Step 2 - CO2
# ---------------------------------------------------------------------------
st.header("Step 2 - CO2")

fig_co2 = go.Figure()
fig_co2.add_trace(line_trace(plot_df["t"], plot_df["chi_co2"], "chi_co2 (raw)", "blue"))
fig_co2.add_trace(
    line_trace(plot_df["t"], plot_df["chi_pred_co2"], "chi_pred_co2 (fit)", "red", dash="dash")
)
fig_co2.update_layout(
    xaxis_title="t (s)", yaxis_title="chi_co2", height=400, legend=dict(orientation="h")
)
st.plotly_chart(fig_co2, width='stretch')

co2_metric_cols = st.columns(5)
co2_metric_cols[0].metric("f_co2", f"{first_row['f_co2']:.4g}")
co2_metric_cols[1].metric("r2_f_co2", f"{first_row['r2_f_co2']:.4g}")
co2_metric_cols[2].metric("rmse_f_co2", f"{first_row['rmse_f_co2']:.4g}")
co2_metric_cols[3].metric("sigma_f_co2", f"{first_row['sigma_f_co2']:.4g}")
co2_metric_cols[4].metric("linear", str(first_row["linear"]))

st.divider()

# ---------------------------------------------------------------------------
# Step 3 - Additional gases
# ---------------------------------------------------------------------------
st.header("Step 3 - Additional gases")

gas_col1, gas_col2 = st.columns(2)

with gas_col1:
    fig_n2o = go.Figure()
    fig_n2o.add_trace(line_trace(plot_df["t"], plot_df["chi_n2o"], "chi_n2o (raw)", "green"))
    fig_n2o.add_trace(
        line_trace(plot_df["t"], plot_df["chi_pred_n2o"], "chi_pred_n2o (fit)", "green", dash="dash")
    )
    fig_n2o.update_layout(
        xaxis_title="t (s)", yaxis_title="chi_n2o", height=350, legend=dict(orientation="h")
    )
    st.plotly_chart(fig_n2o, width='stretch')

    n2o_metric_cols = st.columns(4)
    n2o_metric_cols[0].metric("f_n2o", f"{first_row['f_n2o']:.4g}")
    n2o_metric_cols[1].metric("r2_f_n2o", f"{first_row['r2_f_n2o']:.4g}")
    n2o_metric_cols[2].metric("rmse_f_n2o", f"{first_row['rmse_f_n2o']:.4g}")
    n2o_metric_cols[3].metric("sigma_f_n2o", f"{first_row['sigma_f_n2o']:.4g}")

with gas_col2:
    fig_ch4 = go.Figure()
    fig_ch4.add_trace(line_trace(plot_df["t"], plot_df["chi_ch4"], "chi_ch4 (raw)", "orange"))
    fig_ch4.add_trace(
        line_trace(plot_df["t"], plot_df["chi_pred_ch4"], "chi_pred_ch4 (fit)", "orange", dash="dash")
    )
    fig_ch4.update_layout(
        xaxis_title="t (s)", yaxis_title="chi_ch4", height=350, legend=dict(orientation="h")
    )
    st.plotly_chart(fig_ch4, width='stretch')

    ch4_metric_cols = st.columns(4)
    ch4_metric_cols[0].metric("f_ch4", f"{first_row['f_ch4']:.4g}")
    ch4_metric_cols[1].metric("r2_f_ch4", f"{first_row['r2_f_ch4']:.4g}")
    ch4_metric_cols[2].metric("rmse_f_ch4", f"{first_row['rmse_f_ch4']:.4g}")
    ch4_metric_cols[3].metric("sigma_f_ch4", f"{first_row['sigma_f_ch4']:.4g}")

st.divider()

# ---------------------------------------------------------------------------
# Step 4 - Chamber conditions
# ---------------------------------------------------------------------------
st.header("Step 4 - Chamber conditions")

fig_chamber = go.Figure()
fig_chamber.add_trace(
    go.Scatter(x=group["t"], y=group["TA"], mode="lines", name="TA", line=dict(color="firebrick"))
)
fig_chamber.add_trace(
    go.Scatter(
        x=group["t"], y=group["T_cavity"], mode="lines", name="T_cavity", line=dict(color="darkorange")
    )
)
fig_chamber.add_trace(
    go.Scatter(
        x=group["t"],
        y=group["P_cavity"],
        mode="lines",
        name="P_cavity",
        line=dict(color="steelblue"),
        yaxis="y2",
    )
)
fig_chamber.update_layout(
    xaxis_title="t (s)",
    yaxis=dict(title="Temperature"),
    yaxis2=dict(title="P_cavity", overlaying="y", side="right"),
    height=400,
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_chamber, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
st.header("Labelling")

existing_codes_str = labels.get(current_mmnt_id, "")
existing_codes = existing_codes_str.split("|") if existing_codes_str else []

selected_codes = st.multiselect(
    "Reason codes (select PASS if this measurement is clean)",
    options=REASON_CODES,
    default=existing_codes,
    key=f"reason_codes_{current_mmnt_id}",
)

if st.button("Save Label", type="primary", width='stretch'):
    labels[current_mmnt_id] = "|".join(selected_codes)
    save_labels(labels)
    st.success(f"Saved label for {current_mmnt_id}")
