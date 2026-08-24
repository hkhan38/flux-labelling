# Flux QC Labelling App

A Streamlit app for visually inspecting chamber flux measurements and
assigning QC reason codes, used to train a machine learning model.

## Setup

Requires Python 3 installed (`python3 --version` to check).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Adding data

Create a `raw/` folder next to `app.py` (if it doesn't already exist) and
copy your raw flux export file(s) into it. Any filename works (e.g.
`dt_flux_2024-07-07.csv`, `dt_flux_2024-07-08.csv`), and you can drop in
multiple files at once — they're all loaded and combined automatically.
Each file just needs the same columns as the existing exports.

## Running

```bash
streamlit run app.py
```

This works from any directory — the app always looks for `raw/` and writes
`output/` next to `app.py` itself, not wherever you happen to run the
command from.

## Usage

- Use **Previous** / **Next** to move between measurements, or **Skip to
  next unlabelled** to jump ahead.
- Inspect the CO2, N2O/CH4, and chamber condition charts for each
  measurement.
- Select any applicable reason codes (select **PASS** if the measurement
  looks clean) and click **Save Label**.
- Labels are written to `./output/flux_labels.csv` and are reloaded
  automatically the next time the app starts, so progress is never lost.
