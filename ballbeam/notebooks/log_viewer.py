import json
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="BallBeam Log Viewer", layout="wide")


st.title("📈 BallBeam Log Viewer")


# --- Path setup ---
from ballbeam.paths import ROOT_PATH

data_dir = ROOT_PATH / "data"

subdirs = [str(x.relative_to(data_dir)) for x in data_dir.glob("**/*") if x.is_dir()]
subdir = st.selectbox("Data directory", options=["", *subdirs])

if subdir:
    data_dir = data_dir / subdir


# File selection
log_files = sorted(data_dir.glob("*.json"))
if not log_files:
    st.error("No JSON log files found in the data directory.")
    st.stop()

selected_file = st.selectbox("Select a log file", log_files, format_func=lambda p: p.name)


# --- Load and preprocess data ---
@st.cache_data
def load_log(path: Path):
    with path.open() as f:
        log_data = json.load(f)
    df = pd.DataFrame(log_data)

    # Time since start
    df["time_since_start"] = df["time_now"] - df["time_now"].min()

    # Expand the state estimates
    state_names = ["position_error", "velocity_error", "position_error_sum", "action"]
    state_estimate_column_names = [f"state_estimate__{s}" for s in state_names]
    df[state_estimate_column_names] = df["state_estimate"].apply(pd.Series)

    # Infer absolute position
    df["state_estimate__position"] = df["state_estimate__position_error"] + df["setpoint"]

    return df


df = load_log(selected_file)

st.success(f"Loaded {selected_file.name} ({len(df)} entries)")

# --- Plot
st.subheader("Position vs. Time")
fig = px.line(
    df,
    x="time_since_start",
    y=["observation", "state_estimate__position", "setpoint"],
    title="Position Tracking",
    labels={"time_since_start": "Time Since Start (s)", "value": "Position"},
)
st.plotly_chart(fig, width="stretch")

st.subheader("Action vs. Time")
fig = px.line(
    df,
    x="time_since_start",
    y=["setpoint", "action"],
    title="Control Action",
    labels={"time_since_start": "Time Since Start (s)", "value": "Action"},
)
st.plotly_chart(fig, width="stretch")

# --- Data preview ---
with st.expander("Show raw data"):
    st.dataframe(df)
