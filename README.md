# ☀️ Photovoltaic Power Forecasting using Hybrid Model

Forecast solar or photovoltaic power output using an end-to-end, reproducible machine learning pipeline combining traditional regression, deep learning (LSTM), and hybrid modeling approaches.

---

## 🚀 Features

- Flexible data preprocessing and feature engineering
- Supports Random Forest, LSTM, and hybrid ensemble modeling
- Detailed evaluation: MAE, RMSE, (Adjusted) R², and per-epoch error tracking
- Modular source code for clarity and maintainability
- Jupyter/Colab notebook for step-by-step demonstration
- Well-documented datasets and sample files for quick starts

---

## 📂 Project Structure

```
Photovoltaic-Power-Forecasting-using-Hybrid-Model/
├── data/           # Organized datasets (train/test/raw) + README
├── notebook/       # Jupyter/Colab notebooks (main workflow)
├── requirements/   # Python dependencies (requirements.txt, README)
├── results/        # Model metrics (results.json), key error plots, README
├── samples/        # Example input/output files for quick testing, README
├── src/            # Core code: preprocessing, feature engineering, training, utils
├── .gitignore
├── LICENSE
├── README.md
```

---

## 🛠️ Requirements

Install all dependencies in a fresh environment:

```
pip install -r requirements/requirements.txt
```

---

---

## 📊 Results

See [`results/`](results/) for:
- Summary table (`results.json`): adjusted R² for Random Forest, LSTM, hybrid models
- Convergence/error plots: `mae_vs_epoch.png`, `rmse_vs_epoch.png`, `r2_vs_epoch.png`
- Example result:  
  | Model        | Adjusted R²     |
  |--------------|-----------------|
  | RandomForest | 0.99976         |
  | LSTM         | 0.99869         |
  | HybridModel  | 0.99970         |

---

## 📁 Data

See [`data/README.md`](data/README.md) for folder and file organization, sample columns, and preparation notes.  
_(**Note:** Large/full datasets are not tracked in git. Use provided samples for testing.)_

---

## 📌 Folder Guide

- `data/`         — All datasets (train, test, raw, combined)
- `notebook/`     — Demo notebook (with Colab link)
- `requirements/` — Dependency file(s)
- `results/`      — Metrics, plots, and performance summaries
- `samples/`      — Example input/output files
- `src/`          — Modular Python code for the ML pipeline

---

## 📝 License

Distributed under the MIT License.

---

## 👤 Author

**[Kamal-Shirupa](https://github.com/Kamal-Shirupa)**  
Contributions and feedback are welcome!  
Create an issue or a pull request for suggestions, improvements, or questions.

---
