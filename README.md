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

## Running

```bash
streamlit run app.py
```

## Usage

- Upload your raw flux export CSV under **Upload data file** to begin. If
  you've labelled some of this data before, also upload the
  `flux_labels.csv` you previously downloaded under **Upload existing
  labels file** to pick up where you left off.
- Use **Previous** / **Next** to move between measurements, or **Skip to
  next unlabelled** to jump ahead.
- Inspect the CO2, N2O/CH4, and chamber condition charts for each
  measurement.
- Select any applicable reason codes (select **PASS** if the measurement
  looks clean) and click **Save Label**.
- Labels are kept in the browser session only — nothing is saved to disk.
  Click **Download Labels** regularly to save your progress as
  `flux_labels.csv`; if you close or refresh the browser without
  downloading, unsaved labels are lost.
