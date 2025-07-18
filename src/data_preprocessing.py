import pandas as pd

def clean_dataframe(df):
    """Standardize column names and impute missing values with column means."""
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def load_and_prepare_data(horizon_path, irradiance_path, panelweather_path):
    horizon_df = pd.read_csv(horizon_path)
    irradiance_df = pd.read_csv(irradiance_path)
    panelweather_df = pd.read_csv(panelweather_path)

    # Clean
    horizon_df = clean_dataframe(horizon_df)
    irradiance_df = clean_dataframe(irradiance_df)
    panelweather_df = clean_dataframe(panelweather_df)

    # Parse date/time
    irradiance_df['time'] = pd.to_datetime(irradiance_df['time'], format='%Y%m%d:%H%M')
    panelweather_df['time(UTC)'] = pd.to_datetime(panelweather_df['time(UTC)'], format='%Y%m%d:%H%M')

    # Merge
    merged_df = pd.merge(irradiance_df, panelweather_df, left_on='time', right_on='time(UTC)', how='outer')
    return horizon_df, irradiance_df, panelweather_df, merged_df
