# BIG_DATA_ASSIGNMENT

End-to-end pipeline for **stock portfolio construction** using both **quantitative indicators** (derived from price/volume) and **textual sentiment** (from financial news). The project builds daily multimodal time series, trains temporal models (LSTM, Transformer, 1D-CNN), and forms a top-k, equal-weight portfolio from predicted returns. Experiments are organized in `main.ipynb`.

---

## Repository Structure

```text
BIG_DATA_ASSIGNMENT/
├── data/                 # Local data cache (price/news CSVs)
├── model/                # Model definitions (1D-CNN / LSTM / Transformer, training functions)
├── utils/                # Data loaders, indicators, feature engineering helpers
├── main.ipynb            # End-to-end experiment notebook
└── README.md             # You are here
```

---

## Environment Setup (venv)

The project uses Python 3.9–3.11 (tested with 3.10). Create a clean virtual environment and install dependencies.

```bash
# 0) (Optional) clone
git clone https://github.com/GuGu97/BIG_DATA_ASSIGNMENT.git
cd BIG_DATA_ASSIGNMENT

# 1) Create & activate virtual environment
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install dependencies
# If requirements.txt exists:
pip install -r requirements.txt

# Otherwise install common deps used in the project:
pip install numpy pandas matplotlib scikit-learn torch torchvision torchaudio jupyter ipykernel python-dotenv requests tabulate
python -m ipykernel install --user --name big-data-venv
```
---

## How to Run

All experiments, including data preprocessing, model training, evaluation, and final results, are documented in **`main.ipynb`**.  
Please open this notebook to follow the complete workflow and reproduce the results step by step.