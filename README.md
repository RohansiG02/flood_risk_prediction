## Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Dataset summary and headline results |
| Model comparison | Bagging vs boosting vs stacking: PR-AUC, F1, confidence intervals, new-alert detection |
| Station view | Water level vs official thresholds, and each model's predicted alert probability |
| Level forecast | Water level regression results, including the RF + ANN hybrids |
| RF + ARIMA | Time-series hybrid results for each station |
| Feature importance | Which inputs drive the predictions |
| What-if analysis | Change the water level and rainfall and see a trained model's prediction |
| Trained models | Load the saved models, score them on the test set and download predictions |

## Run it locally

Requirements: Python 3.9 or newer.

### 1. Get the code

```bash
git clone https://github.com/RohansiG02/flood-risk-dashboard.git
cd flood-risk-dashboard
```

### 2. Create an environment and install the packages (first time only)

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 3. Add the trained models (optional)

Download `model.zip` from Google Drive (see [Trained models](#trained-models-download-separately)) and unzip it into the
`models/` folder. Without this step everything works except the **Trained models** page and the model choice on the
**What-if analysis** page.

### 4. Start the dashboard

macOS / Linux:

```bash
.venv/bin/streamlit run app.py
```

Windows:

```bat
.venv\Scripts\streamlit run app.py
```

Open **http://localhost:8501** in your browser. Press **Ctrl + C** in the terminal to stop the dashboard.

From the second time onwards, only step 4 is needed. On macOS or Linux, `./run_dashboard.sh` runs steps 2 and 4 for you.

**"Port 8501 is already in use"?** Another copy is still running. Stop it with `pkill -f "streamlit run"`
(macOS / Linux), or start on another port: `.venv/bin/streamlit run app.py --server.port 8502`.

## Trained models (download separately)

The trained models are about 2 GB, which is too large for GitHub, so they are stored on Google Drive:

**[https://drive.google.com/drive/folders/1NeUGE5EKTUEzpwkt8HMZX70-MLxlu5Ph?usp=drive_link](https://drive.google.com/file/d/155ihBRejEXY5qknzUBuQJG0MF1f7ZduK/view?usp=sharing)**

Download `model.zip` and unzip it into the `models/` folder, so it looks like this:

```
models/
├── models_clf/   12 classifiers
├── models_reg/   15 regressors (including the RF + ANN hybrids)
└── models_ts/    rf_arima_hybrid.joblib
```

The dashboard works **without** the models. Every results page reads the small files in `data/`.
The models are only needed for the **Trained models** page and for choosing a model on the **What-if analysis** page.
Without them, the What-if page uses the CatBoost models included in `data/`.

## Project structure

```
├── app.py               Streamlit dashboard
├── trained_models.py    Loads the trained models and prepares their inputs
├── requirements.txt     Python packages
├── run_dashboard.sh     Start script
├── .streamlit/          Theme (dark, blue accent)
├── assets/              Logo
├── data/                Results, predictions and station data used by the dashboard
├── model_inputs/        Test-set inputs and preprocessing objects for the trained models
└── models/              Trained models (download from Google Drive, not in the repository)
```
