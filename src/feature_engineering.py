def add_time_features(df, time_column='time'):
    df['hour'] = df[time_column].dt.hour
    df['day'] = df[time_column].dt.day
    df['month'] = df[time_column].dt.month
    return df

def select_features(df, target_col, drop_columns=None, n_features=10):
    if drop_columns is None:
        drop_columns = []
    drop_columns = [col for col in drop_columns if col in df.columns]
    features = df.drop(columns=drop_columns + [target_col])
    y = df[target_col]
    features = features.iloc[:, :n_features]  # Select first N features.
    return features, y
