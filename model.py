from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def train_delay_prediction_model(df):
    features = [
        'hour', 'weekday', 'is_peak', 'trip_hour',
        'adt', 'peaktraffic', 'pccar', 'pclcv',
        'STOPLAT', 'STOPLON', 'NS_SPEED_LIMIT', 'Shape__Length',
        'ROUTENAME_encoded', 'MODE_encoded',
        'avg_loc_delay', 'prev_stop_encoded', 'next_stop_encoded',
        'hour_bin_morning', 'hour_bin_afternoon', 'hour_bin_evening', 'hour_bin_night',
        'minute', 'second'
    ]
    
    target = 'delay_seconds'

    df_model = df[features + [target]].dropna()
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model