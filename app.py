# -*- coding: utf-8 -*-
"""Streamlit app for Airbnb NYC price prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Airbnb NYC Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #B0B0B0;
        margin-bottom: 30px;
    }

    .section-card {
        background-color: #1E1E2F;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #33334D;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
        margin-bottom: 20px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #0F9D58, #0B6E3D);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 25px rgba(0,0,0,0.35);
    }

    .prediction-price {
        font-size: 46px;
        font-weight: 800;
        margin-top: 10px;
    }

    .small-text {
        color: #B0B0B0;
        font-size: 14px;
    }

    div.stButton > button {
        width: 100%;
        height: 3em;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
    }

    /* Make dropdown and input text clearer */
    div[data-baseweb="select"] * {
        font-size: 15px !important;
    }

    div[data-baseweb="input"] input {
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model bundle
# -----------------------------
bundle = joblib.load("airbnb_price_model.pkl")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
neighbourhood_price_map = bundle["neighbourhood_price_map"]

# Load average coordinates for each neighbourhood.
# This lets the app hide latitude and longitude inputs from users.
neighbourhood_coord_map = bundle.get("neighbourhood_coord_map", {})

# Load valid neighbourhoods for each neighbourhood group
neighbourhood_group_map = bundle.get("neighbourhood_group_map", {})

default_neighbourhood_encoded = bundle["default_neighbourhood_encoded"]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🏠 Airbnb Price Predictor")
    st.write(
        "This app estimates Airbnb listing prices in New York City using a trained machine learning model."
    )

    st.markdown("---")

    st.subheader("Model Input Features")
    st.write(
        """
        The prediction uses listing information such as:
        - Location
        - Neighbourhood group
        - Room type
        - Minimum nights
        - Review activity
        - Availability
        - Host listing count
        """
    )

    st.markdown("---")
    st.caption("Developed for AML Group Project")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Group 2 Airbnb NYC Price Prediction Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Estimate Airbnb listing prices based on property, location and review activity.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Main layout
# -----------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📍 Listing Information")

    col1, col2 = st.columns(2)

    with col1:
        neighbourhood_group = st.selectbox(
            "Neighbourhood Group",
            ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
        )

        # Show only neighbourhoods that belong to the selected neighbourhood group.
        # If the model bundle does not contain neighbourhood_group_map, use all neighbourhoods as a fallback.
        neighbourhood_options = neighbourhood_group_map.get(
            neighbourhood_group,
            sorted(neighbourhood_price_map.keys())
        )

        neighbourhood = st.selectbox(
            "Neighbourhood",
            neighbourhood_options
        )

        room_type = st.selectbox(
            "Room Type",
            ["Entire home/apt", "Private room", "Shared room"]
        )

        minimum_nights = st.number_input(
            "Minimum Nights",
            min_value=1,
            max_value=365,
            value=1,
            step=1,
            help="Minimum nights must be between 1 and 365."
        )

    with col2:
        availability_365 = st.number_input(
            "Availability Per Year",
            min_value=0,
            max_value=365,
            value=100,
            step=1,
            help="Availability must be between 0 and 365 days."
        )

        calculated_host_listings_count = st.number_input(
            "Host Listings Count",
            min_value=1,
            max_value=500,
            value=1,
            step=1,
            help="Host listing count must be between 1 and 500."
        )

    # Automatically assign latitude and longitude based on the selected neighbourhood.
    # The model still receives latitude and longitude, but the user does not need to enter them.
    coords = neighbourhood_coord_map.get(
        neighbourhood,
        {"latitude": 40.7128, "longitude": -74.0060}
    )

    latitude = coords["latitude"]
    longitude = coords["longitude"]

    st.caption("Latitude and longitude are automatically assigned from the selected neighbourhood.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("⭐ Review Information")

    review_col1, review_col2 = st.columns(2)

    with review_col1:
        number_of_reviews = st.number_input(
            "Number of Reviews",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            help="Number of reviews must be between 0 and 1000."
        )

    with review_col2:
        reviews_per_month = st.number_input(
            "Reviews Per Month",
            min_value=0.0,
            max_value=60.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            help="Reviews per month must be between 0.00 and 60.00."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    predict_button = st.button("Predict Airbnb Price")

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🏘️ Listing Preview")

    st.markdown(
        f"""
        <p class="small-text">
        <b>Neighbourhood Group:</b> {neighbourhood_group}<br>
        <b>Neighbourhood:</b> {neighbourhood}<br>
        <b>Room Type:</b> {room_type}<br>
        <b>Coordinates:</b> Automatically assigned from neighbourhood<br>
        <b>Availability:</b> {availability_365} days per year
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.write(
        "This preview summarizes the selected Airbnb listing information before prediction."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction logic
# -----------------------------
if predict_button:

    validation_errors = []

    if minimum_nights < 1 or minimum_nights > 365:
        validation_errors.append("Minimum nights must be between 1 and 365.")

    if number_of_reviews < 0 or number_of_reviews > 1000:
        validation_errors.append("Number of reviews must be between 0 and 1000.")

    if reviews_per_month < 0 or reviews_per_month > 60:
        validation_errors.append("Reviews per month must be between 0 and 60.")

    if calculated_host_listings_count < 1 or calculated_host_listings_count > 500:
        validation_errors.append("Host listings count must be between 1 and 500.")

    if availability_365 < 0 or availability_365 > 365:
        validation_errors.append("Availability must be between 0 and 365.")

    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()

    input_data = {col: 0 for col in feature_columns}

    # Numerical features
    input_data["latitude"] = latitude
    input_data["longitude"] = longitude
    input_data["availability_365"] = availability_365

    # Log-transformed features
    input_data["minimum_nights_log"] = np.log1p(minimum_nights)
    input_data["number_of_reviews_log"] = np.log1p(number_of_reviews)
    input_data["reviews_per_month_log"] = np.log1p(reviews_per_month)
    input_data["host_listings_log"] = np.log1p(calculated_host_listings_count)

    # Feature engineering
    input_data["with_reviews"] = 1 if number_of_reviews > 0 else 0

    # One-hot encode room type
    room_col = f"room_type_{room_type}"
    if room_col in input_data:
        input_data[room_col] = 1

    # One-hot encode neighbourhood group
    group_col = f"neighbourhood_group_{neighbourhood_group}"
    if group_col in input_data:
        input_data[group_col] = 1

    # Target encode neighbourhood
    input_data["neighbourhood_encoded"] = neighbourhood_price_map.get(
        neighbourhood,
        default_neighbourhood_encoded
    )

    # Convert to DataFrame in the same feature order as training
    input_df = pd.DataFrame([input_data])[feature_columns]

    # Predict log price
    predicted_log_price = model.predict(input_df)[0]

    # Convert back to actual price
    predicted_price = np.expm1(predicted_log_price)

    st.markdown("---")

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div>Estimated Airbnb Listing Price</div>
                <div class="prediction-price">${predicted_price:.2f}</div>
                <div>per night</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col2:
        st.subheader("📋 Listing Summary")

        summary_df = pd.DataFrame({
            "Feature": [
                "Neighbourhood Group",
                "Neighbourhood",
                "Room Type",
                "Minimum Nights",
                "Number of Reviews",
                "Reviews Per Month",
                "Availability Per Year",
                "Host Listings Count"
            ],
            "Selected Value": [
                neighbourhood_group,
                neighbourhood,
                room_type,
                minimum_nights,
                number_of_reviews,
                reviews_per_month,
                availability_365,
                calculated_host_listings_count
            ]
        })

        st.dataframe(summary_df, use_container_width=True)

    st.info(
        "Note: This is a machine learning estimate based on historical Airbnb NYC listing data. "
        "The predicted price should be treated as a decision-support value, not a guaranteed market price."
    )
