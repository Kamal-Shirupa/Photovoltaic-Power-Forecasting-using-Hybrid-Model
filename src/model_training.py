import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from scikeras.wrappers import KerasRegressor

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_preprocessor(feature_names):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, feature_names)
    ])
    return preprocessor

def train_and_evaluate(model, model_name, preprocessor, X_train, X_test, y_train, y_test, is_lstm=False):
    if is_lstm:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=32, verbose=1)
        y_pred = model.predict(X_test)
    else:
        pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} - MAE: {mae}\n{model_name} - RMSE: {rmse}\n{model_name} - R2: {r2}')
    return model, mae, rmse, r2, y_pred

def train_test_split_and_impute(features, y, test_size=0.2, random_state=42):
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    y_imputed = y.fillna(y.mean())
    X_train, X_test, y_train, y_test = train_test_split(
        features_imputed, y_imputed, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
