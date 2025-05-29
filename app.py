# app.py

import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta
from create_df import create_df
from model import train_delay_prediction_model
from predict import predict_delay_from_user_input, get_possible_end_stops

# Ensure set_page_config is the first Streamlit command
st.set_page_config(
    page_title="Public Transport Congestion Prediction System",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache Data Loading and Model Training
@st.cache_data(show_spinner="Loading and preprocessing data, please wait....")
def load_and_prepare():
    df = create_df()
    model = train_delay_prediction_model(df)
    return df, model

df, model = load_and_prepare()


# Retrieve all stop names (for dropdown selection)
valid_start_stops = df[df['next_stop_id'].notna()]['stop_name'].dropna().str.strip().str.upper().unique()
stop_names = sorted(valid_start_stops)


# Page Content
st.title("Public Transport Congestion Prediction and Travel Advice")
st.markdown(" Welcome to the system！")
st.markdown("---")

# Sidebar user input
st.sidebar.header("Please enter your travel information")

# Starting point: Pull down + input
default_stop = "PUKEKOHE TRAIN STATION 1"
default_index = stop_names.index(default_stop) if default_stop in stop_names else 0

selected_start = st.sidebar.selectbox("🔽 Select departure stop", stop_names, index=default_index)


start_stop =  selected_start

# Input time
user_time_obj = st.sidebar.time_input("⏰ Departure time", value=pd.to_datetime("13:00:00").time())

user_time_str = user_time_obj.strftime("%H:%M:%S")



# Destination: Depends on the starting point and provides candidate stations
possible_ends = []
if user_time_str and start_stop:
    possible_ends = get_possible_end_stops(df, user_time_str, start_stop)

if possible_ends:
    st.sidebar.success(f"🔍  A total of {len(possible_ends)} reachable destination stops found:")
    selected_end = st.sidebar.selectbox("🔽 Select destination stop", possible_ends)
    end_stop =  selected_end

    # Legality Check
    if end_stop not in possible_ends:
        st.sidebar.error("The selected destination stop is not in the list of reachable stops from the chosen departure stop. Please select or enter again.")
        submit = False
    else:
        submit = st.sidebar.button("Start Prediction")
else:
    st.sidebar.warning("Unable to recommend destination stops for the selected departure stop and time")
    submit = False


if submit:
    with st.spinner("Predicting delay..."):
        try:
            result = predict_delay_from_user_input(df, model, user_time_str, start_stop, end_stop)
            match = re.search(r"(-?\d+\.?\d*)", result)
            delay_seconds = float(match.group(1)) if match else 0

            user_departure = datetime.combine(pd.Timestamp.today(), user_time_obj)

            scheduled_arrival_time = datetime.combine(pd.Timestamp.today(), datetime.strptime("17:00:00", "%H:%M:%S").time())

            predicted_arrival_time = scheduled_arrival_time + timedelta(seconds=delay_seconds)

            total_travel_minutes = (predicted_arrival_time - user_departure).total_seconds() / 60

            st.success(result)

            if total_travel_minutes > 90:
                st.warning("Suggestion: The predicted wait time for your trip is too long. Please consider departing closer to the estimated time or choosing an alternative route.")
            elif total_travel_minutes > 60:
                st.info("Suggestion: The predicted arrival time is still a while away, so there's no need to worry about your departure time.")
            else:
                st.success("Suggestion: The predicted vehicle is arriving soon. Please get ready to depart.")
        except Exception as e:
            st.error(f"An error occurred during prediction:{e}")

    