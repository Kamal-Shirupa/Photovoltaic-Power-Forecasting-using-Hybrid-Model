# Data (`data/`)

This folder contains all datasets and input files used for building and evaluating the solar power forecasting models.

## Description

- `panelandweather_*` files contain features such as temperature, humidity, irradiance, wind speed, wind direction, and solar power output.
- `irradiance_*` files contain solar irradiance measurements alongside other weather variables.
- Files are timestamped using `time(UTC)` as the primary time index.

## Structure

```
data/
├── train/                 # Training datasets
│   ├── panelandweather_train.csv
│   └── irradiance_train.csv
├── test/                  # Testing datasets
│   ├── panelandweather_test.csv
│   └── irradiance_test.csv
├── datasets/              # Combined or original datasets
│   ├── panelandweather.csv
│   └── irradiance.csv
│   └── horizon.csv
```

## Notes

- All datasets are in comma-separated `.csv` format.
- Ensure consistent date formatting when merging or preprocessing files.
- Use the `train/` and `test/` folders during model development for proper evaluation.
