import pandas as pd
from datetime import datetime, timedelta

# Match the trip_id that conforms to the order of time and site
def find_candidate_trips(df, user_time_str, start_stop_name, end_stop_name):
    user_time_obj = pd.to_datetime(user_time_str, format="%H:%M:%S").time()
    start_stop_name = start_stop_name.strip().upper()
    end_stop_name = end_stop_name.strip().upper()

    candidates = df[
        (df['stop_name'].str.strip().str.upper() == start_stop_name) &
        (pd.to_datetime(df['scheduled_time'], format="%H:%M:%S", errors='coerce').dt.time >= user_time_obj)
    ]

    candidate_trip_ids = candidates['trip_id'].unique()
    matched_trips = []

    for trip_id in candidate_trip_ids:
        trip_df = df[df['trip_id'] == trip_id].sort_values('scheduled_time')
        stops_sequence = trip_df['stop_name'].str.strip().str.upper().tolist()

        if start_stop_name in stops_sequence and end_stop_name in stops_sequence:
            start_index = stops_sequence.index(start_stop_name)
            end_index = stops_sequence.index(end_stop_name)
            if end_index > start_index:
                matched_trips.append(trip_id)

    return matched_trips


# Given trip_id and the names of the start and end points, track the complete stop_id path
def trace_stop_ids(df, trip_id, start_stop_name, end_stop_name):
    trip_df = df[df['trip_id'] == trip_id]
    start_stop_name = start_stop_name.strip().upper()
    end_stop_name = end_stop_name.strip().upper()

    start_row = trip_df[trip_df['stop_name'].str.strip().str.upper() == start_stop_name]
    if start_row.empty:
        return []

    current_stop_id = start_row.iloc[0]['stop_id']
    path = [current_stop_id]
    visited = set(path)

    while True:
        row = trip_df[trip_df['stop_id'] == current_stop_id]
        if row.empty:
            break

        stop_name = row.iloc[0]['stop_name'].strip().upper()
        if stop_name == end_stop_name:
            break

        next_stop_id = row.iloc[0]['next_stop_id']
        if pd.isna(next_stop_id) or next_stop_id in visited:
            break

        path.append(next_stop_id)
        visited.add(next_stop_id)
        current_stop_id = next_stop_id

    return path


# Extract features from trip_id + stop_ids
def extract_features_for_prediction(df, trip_id, stop_ids, user_time_str):
    today = pd.Timestamp.today().normalize()
    full_datetime = pd.to_datetime(f"{today.date()} {user_time_str}")
    hour = full_datetime.hour
    minute = full_datetime.minute
    second = full_datetime.second
    weekday = full_datetime.weekday()
    is_peak = int(hour in range(7, 10) or hour in range(16, 19))

    first_stop_id = stop_ids[0]
    row = df[(df['trip_id'] == trip_id) & (df['stop_id'] == first_stop_id)]
    if row.empty:
        raise ValueError("❌ Unable to find data in the DataFrame for the specified trip_id and stop_id")
    row = row.iloc[0]

    return pd.DataFrame([{
        'hour': hour,
        'weekday': weekday,
        'is_peak': is_peak,
        'trip_hour': row['trip_hour'],
        'adt': row['adt'],
        'peaktraffic': row['peaktraffic'],
        'pccar': row['pccar'],
        'pclcv': row['pclcv'],
        'STOPLAT': row['STOPLAT'],
        'STOPLON': row['STOPLON'],
        'NS_SPEED_LIMIT': row['NS_SPEED_LIMIT'],
        'Shape__Length': row['Shape__Length'],
        'ROUTENAME_encoded': row['ROUTENAME_encoded'],
        'MODE_encoded': row['MODE_encoded'],
        'avg_loc_delay': row['avg_loc_delay'],
        'prev_stop_encoded': row['prev_stop_encoded'],
        'next_stop_encoded': row['next_stop_encoded'],
        'hour_bin_morning': int(hour in range(6, 10)),
        'hour_bin_afternoon': int(hour in range(12, 16)),
        'hour_bin_evening': int(hour in range(16, 20)),
        'hour_bin_night': int(hour in range(20, 24)),
        'minute': minute,
        'second': second,
    }])


# Complete the entire prediction process starting from user input
def predict_delay_from_user_input(df, model, user_time_str, start_stop_name, end_stop_name):
    matched_trip_ids = find_candidate_trips(df, user_time_str, start_stop_name, end_stop_name)
    if not matched_trip_ids:
        return "❌ No vehicle found that matches the specified time and stop sequence."

    trip_id = matched_trip_ids[0]
    stop_ids = trace_stop_ids(df, trip_id, start_stop_name, end_stop_name)
    if not stop_ids:
        return "❌ Unable to trace the complete stop path from the departure to the destination stop."

    test_input = extract_features_for_prediction(df, trip_id, stop_ids, user_time_str)
    predicted_delay = model.predict(test_input)[0]

    rows = df[(df['trip_id'] == trip_id) & (df['stop_id'].isin(stop_ids))].copy()

    arrival_time_str = "Null"
    actual_arrival_str = "Null"

    if not rows.empty:
        today = pd.Timestamp.today().normalize()
        user_time_full = pd.to_datetime(f"{today.date()} {user_time_str}").tz_localize(None)
        rows['arrival_dt'] = pd.to_datetime(rows['arrival_dt'], errors='coerce').dt.tz_localize(None)

        rows = rows.dropna(subset=['arrival_dt'])

        if not rows.empty:
            rows['time_diff'] = (rows['arrival_dt'] - user_time_full).abs()
            best_row = rows.loc[rows['time_diff'].idxmin()]

            arrival_dt = best_row['arrival_dt']
            arrival_time_str = arrival_dt.time().strftime('%H:%M:%S')

            actual_arrival = arrival_dt + pd.to_timedelta(predicted_delay, unit='s')
            actual_arrival_str = actual_arrival.time().strftime('%H:%M:%S')


    delay_type = "Delay" if predicted_delay >= 0 else "Early Arrival"
    delay_value = abs(predicted_delay)

    return f"""
    🚌 **Estimated {delay_type}:** {delay_value:.2f} seconds  
    📍 **Scheduled Arrival Time:** {arrival_time_str}  
    ⏱️ **Estimated Actual Arrival Time:** {actual_arrival_str}
    """


def find_trips_from_start(df, user_time_str, start_stop_name):
    user_time_obj = pd.to_datetime(user_time_str, format="%H:%M:%S").time()
    start_stop_name = start_stop_name.strip().upper()

    candidates = df[
        (df['stop_name'].str.strip().str.upper() == start_stop_name) &
        (pd.to_datetime(df['scheduled_time'], format="%H:%M:%S", errors='coerce').dt.time >= user_time_obj)
    ]

    return candidates['trip_id'].unique().tolist()



def get_possible_end_stops(df, user_time_str, start_stop_name):
    matched_trip_ids = find_trips_from_start(df, user_time_str, start_stop_name)
    if not matched_trip_ids:
        return []

    for trip_id in matched_trip_ids:
        trip_df = df[df['trip_id'] == trip_id].copy()
        trip_df['stop_name'] = trip_df['stop_name'].astype(str).str.strip().str.upper()
        trip_df['stop_id'] = trip_df['stop_id'].astype(str).str.strip()
        trip_df['next_stop_id'] = trip_df['next_stop_id'].astype(str).str.strip()

        start_row = trip_df[trip_df['stop_name'] == start_stop_name.strip().upper()]
        if start_row.empty:
            continue

        current_stop_id = start_row.iloc[0]['stop_id']
        visited = set([current_stop_id])
        possible_ends = []

        while True:
            row = trip_df[trip_df['stop_id'] == current_stop_id]
            if row.empty:
                break

            next_stop_id = row.iloc[0]['next_stop_id']
            if pd.isna(next_stop_id) or next_stop_id in visited:
                break

            next_row = trip_df[trip_df['stop_id'] == next_stop_id]
            if next_row.empty:
                break

            next_stop_name = next_row.iloc[0]['stop_name']
            possible_ends.append(next_stop_name)
            visited.add(next_stop_id)
            current_stop_id = next_stop_id

        if possible_ends:
            return list(dict.fromkeys(possible_ends))

    return []


