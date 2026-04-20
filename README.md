# Electricity demand forecasting (LSTM)

Multivariate LSTM on hourly smart-city energy data to predict the next hour’s electricity demand (MW). Scalers are fit only on the training time range; sequences are split chronologically into train, validation, and test.

## Dataset

- [Smart City Energy Dataset](https://www.kaggle.com/datasets/ayyappanmarimuthu/smart-city-energy-dataset) on Kaggle  
- Default CSV path: `/kaggle/input/smart-city-energy-dataset/smart_city_energy_dataset.csv`

## Project layout

| Path | Role |
|------|------|
| [src/config.py](src/config.py) | Paths, hyperparameters, `FEATURE_COLS` |
| [src/data.py](src/data.py) | Load and sort CSV |
| [src/features.py](src/features.py) | Rolling, lag, cyclical, THI, interactions |
| [src/dataset.py](src/dataset.py) | Fit-on-train scalers, sliding windows, DataLoaders |
| [src/model.py](src/model.py) | LSTM + dropout + head |
| [src/train.py](src/train.py) | Train/val loop, best checkpoint, test + plot |
| [src/evaluate.py](src/evaluate.py) | Metrics and saved figure |
| [scripts/run.py](scripts/run.py) | Local entry: `python scripts/run.py` |
| [notebooks/kaggle_run.ipynb](notebooks/kaggle_run.ipynb) | Clone repo on Kaggle and run `main()` |

Artifacts (checkpoint, plot) go under `artifacts/` (gitignored).

## Run on Kaggle

1. Create a notebook and add the dataset as an input.  
2. Open [notebooks/kaggle_run.ipynb](notebooks/kaggle_run.ipynb), set `REPO_URL` to your fork, run all.  
3. Outputs: `artifacts/best_lstm.pt`, `artifacts/pred_vs_actual.png`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set EDP_CSV_PATH=C:\path\to\smart_city_energy_dataset.csv
python scripts/run.py
```

On Linux/macOS use `export EDP_CSV_PATH=...`.

## Configuration

Environment variables:

- `EDP_CSV_PATH` — path to `smart_city_energy_dataset.csv` (defaults to the Kaggle input path above).

Edit [src/config.py](src/config.py) for sequence length, batch size, hidden size, epochs, and split ratios.

## Model notes

- Inputs: all columns in `FEATURE_COLS` (demand, lags, calendar encodings, weather-derived fields).  
- Target: next-step `electricity_demand_mw` (separate `MinMaxScaler` for y).  
- Training logs **mean** loss per epoch on train and validation; best weights are chosen by validation loss.  
- `ReduceLROnPlateau` reduces the learning rate when validation loss stalls.

## License

MIT — see [LICENSE](LICENSE).
