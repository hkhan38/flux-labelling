# Flux QC Labelling App

A Streamlit app for visually inspecting chamber flux measurements and
assigning QC reason codes, used to train a machine learning model.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

The app loads data from `./raw/dt_flux_2024-07-07.csv`, so run the command
from this folder.

## Usage

- Use **Previous** / **Next** to move between measurements, or **Skip to
  next unlabelled** to jump ahead.
- Inspect the CO2, N2O/CH4, and chamber condition charts for each
  measurement.
- Select any applicable reason codes (select **PASS** if the measurement
  looks clean) and click **Save Label**.
- Labels are written to `./output/flux_labels.csv` and are reloaded
  automatically the next time the app starts, so progress is never lost.
