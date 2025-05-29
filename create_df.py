import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder



def spatial_join_nearest_keepall(left_df, right_df, left_lat, left_lon, right_lat, right_lon, right_cols, k=1):
    
    valid_mask = left_df[left_lat].notna() & left_df[left_lon].notna()
    left_valid = left_df[valid_mask].copy()
    left_coords = np.array(left_valid[[left_lat, left_lon]])

    right_clean = right_df.dropna(subset=[right_lat, right_lon]).copy()
    right_coords = np.array(right_clean[[right_lat, right_lon]])

    available_cols = [col for col in right_cols if col in right_clean.columns]
    missing_cols = set(right_cols) - set(available_cols)
    if missing_cols:
        print("does not exist in the table as follows and will skip the match:", missing_cols)

    nbrs = NearestNeighbors(n_neighbors=k).fit(right_coords)
    distances, indices = nbrs.kneighbors(left_coords)

    matched = right_clean.iloc[indices.flatten()][available_cols].reset_index(drop=True)
    left_valid_result = pd.concat([left_valid.reset_index(drop=True), matched], axis=1)

    result = left_df.copy()

    if available_cols:
        result_valid = result.loc[valid_mask].copy().reset_index(drop=True)

        for col in available_cols:
            result_valid[col] = left_valid_result[col].values

        result.loc[valid_mask, available_cols] = result_valid[available_cols].values
    else:
        print("There are no available columns to write the match")

    return result


def create_df():
    Brc = pd.read_csv('Bus_Route_CLEANED.csv')
    Adc = pd.read_csv('Average_Daily_Traffic_Counts_CLEANED.csv')
    bdr = pd.read_csv('bus_delay_results_c.csv')
    Slc = pd.read_csv('Speed_Limits_CLEANED.csv')
    Bsc = pd.read_csv('Bus_Stop_CLEANED.csv')

    Bsc.rename(columns={"STOPID": "stop_id"}, inplace=True)
    df = bdr.merge(
        Bsc[['stop_id', 'STOPLAT', 'STOPLON', 'MODE', 'PARENTSTATION', 'X', 'Y']],
        on='stop_id',
        how='left'
    )

    stops_txt = pd.read_csv("gtfs/stops.txt", encoding='utf-8', sep=",")

    stops_txt.columns = stops_txt.columns.str.strip().str.lower()

    missing_mask = df['STOPLAT'].isna() | df['STOPLON'].isna()
    df_missing = df[missing_mask]

    df_filled = df_missing.merge(
        stops_txt[['stop_id', 'stop_lat', 'stop_lon', 'stop_name', 'parent_station']],
        on='stop_id',
        how='left'
    )

    df.loc[missing_mask, 'STOPLAT'] = df_filled['stop_lat'].values
    df.loc[missing_mask, 'STOPLON'] = df_filled['stop_lon'].values

    df.loc[missing_mask, 'STOPNAME'] = df_filled.get('stop_name')
    df.loc[missing_mask, 'PARENTSTATION'] = df_filled.get('parent_station')

    Adc = Adc.rename(columns={'X': 'adc_X', 'Y': 'adc_Y'})

    df = spatial_join_nearest_keepall(
        left_df=df,
        right_df=Adc,
        left_lat='Y', left_lon='X',
        right_lat='adc_Y', right_lon='adc_X',
        right_cols=['adt', 'peaktraffic', 'pccar', 'pclcv', 'road_id', 'road_name', 'start_name', 'end_name']
    )

    Slc_max = Slc.groupby('ROAD_ID', as_index=False)['NS_SPEED_LIMIT'].max()

    df = df.merge(
        Slc_max[['ROAD_ID', 'NS_SPEED_LIMIT']],
        left_on='road_id', 
        right_on='ROAD_ID',
        how='left'
    )

    df['ROUTEPATTERN'] = df['trip_id'].str.split('-').str[0]

    df['ROUTEPATTERN'] = df['ROUTEPATTERN'].astype(str)
    Brc['ROUTEPATTERN'] = Brc['ROUTEPATTERN'].astype(str)

    df = df.merge(
        Brc[['ROUTEPATTERN', 'ROUTENAME', 'ROUTENUMBER', 'Shape__Length']],
        on='ROUTEPATTERN',
        how='left'
    )

    df['arrival_time'] = pd.to_datetime(df['arrival_time_scheduled'], errors='coerce')

    df = df.sort_values(['trip_id', 'arrival_time'])

    df['prev_stop_id'] = df.groupby('trip_id')['stop_id'].shift(1)
    df['next_stop_id'] = df.groupby('trip_id')['stop_id'].shift(-1)

    df['delay_minute'] = df['delay_seconds'] / 60
    df['stop_id_group'] = df['stop_id'].str[:4]   
    df['trip_hour'] = pd.to_datetime(df['scheduled_time'], format="%H:%M:%S", errors='coerce').dt.hour

    df['ROUTENAME_encoded'] = LabelEncoder().fit_transform(df['ROUTENAME'].astype(str))
    df['MODE_encoded'] = LabelEncoder().fit_transform(df['MODE'].astype(str))

    df['arrival_dt'] = pd.to_datetime(df['arrival_dt'], errors='coerce')
    df = df.dropna(subset=['arrival_dt'])

    df['hour'] = df['arrival_dt'].dt.hour
    df['weekday'] = df['arrival_dt'].dt.weekday
    df['is_peak'] = df['hour'].between(7, 9) | df['hour'].between(16, 18)
    df['is_noon'] = df['hour'].between(11, 13)
    df['is_afternoon'] = df['hour'].between(14, 17)

    df['hour_bin'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])

    df = pd.get_dummies(df, columns=['hour_bin'])

    df['location_id'] = df['STOPLAT'].astype(str) + '_' + df['STOPLON'].astype(str)
    location_avg_delay = df.groupby('location_id')['delay_seconds'].mean().rename('avg_loc_delay').reset_index()
    df = df.merge(location_avg_delay, on='location_id', how='left')

    df['arrival_dt'] = pd.to_datetime(df['arrival_dt'])

    df['hour'] = df['arrival_dt'].dt.hour
    df['weekday'] = df['arrival_dt'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6])
    df['is_peak'] = df['hour'].between(7, 9) | df['hour'].between(16, 18)

    le_prev = LabelEncoder()
    le_next = LabelEncoder()

    df['prev_stop_encoded'] = le_prev.fit_transform(df['prev_stop_id'].astype(str))
    df['next_stop_encoded'] = le_next.fit_transform(df['next_stop_id'].astype(str))

    df['minute'] = df['arrival_dt'].dt.minute
    df['second'] = df['arrival_dt'].dt.second
    df['adt_peak'] = df['adt'] * df['is_peak']

    df['scheduled_time'] = pd.to_datetime(df['scheduled_time'], format="%H:%M:%S", errors='coerce')
    df['scheduled_time'] = df['scheduled_time'].dt.time

    return df