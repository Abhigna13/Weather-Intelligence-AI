import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Intelligence AI",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🌦️ Weather Intelligence AI")
st.subheader("AI-Powered Temperature Prediction")

st.write(
    "Enter the weather parameters below and get a "
    "temperature prediction from the trained Random Forest model."
)


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8001"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Information")

    st.write("**Backend:** FastAPI")
    st.write("**Model:** RandomForestRegressor")
    st.write("**Prediction:** Temperature")
    st.write("**API Port:** 8001")

    st.divider()

    st.info(
        "Make sure the FastAPI server is running before "
        "using the prediction button."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌡️ Weather Parameters")

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    summary = st.number_input(
        "Summary (Encoded)",
        value=0.0,
        step=1.0
    )

    precip_type = st.number_input(
        "Precipitation Type (Encoded)",
        value=0.0,
        step=1.0
    )

    apparent_temperature = st.number_input(
        "Apparent Temperature (°C)",
        value=18.0,
        step=0.1
    )

    humidity = st.number_input(
        "Humidity",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=14.0,
        step=0.1
    )

    wind_bearing = st.number_input(
        "Wind Bearing (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=225.0,
        step=1.0
    )

    visibility = st.number_input(
        "Visibility (km)",
        min_value=0.0,
        value=12.0,
        step=0.1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    cloud_cover = st.number_input(
        "Cloud Cover",
        value=2.0,
        step=1.0
    )

    pressure = st.number_input(
        "Pressure (millibars)",
        min_value=0.0,
        value=1008.0,
        step=0.1
    )

    daily_summary = st.number_input(
        "Daily Summary (Encoded)",
        value=150.0,
        step=1.0
    )

    year = st.number_input(
        "Year",
        min_value=1900,
        max_value=2100,
        value=2026,
        step=1
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=9,
        step=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=11,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Temperature",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    payload = {

        "summary": summary,

        "precip_type": precip_type,

        "apparent_temperature":
            apparent_temperature,

        "humidity":
            humidity,

        "wind_speed":
            wind_speed,

        "wind_bearing":
            wind_bearing,

        "visibility":
            visibility,

        "cloud_cover":
            cloud_cover,

        "pressure":
            pressure,

        "daily_summary":
            daily_summary,

        "year":
            int(year),

        "month":
            int(month),

        "day":
            int(day)
    }


    try:

        with st.spinner(
            "Running AI temperature prediction..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            predicted_temperature = (
                result[
                    "prediction"
                ][
                    "predicted_temperature_celsius"
                ]
            )

            model_name = (
                result[
                    "model"
                ][
                    "model_name"
                ]
            )

            features_used = (
                result[
                    "model"
                ][
                    "features_used"
                ]
            )


            st.success(
                "Temperature prediction completed successfully."
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.header("🎯 Prediction Result")

            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.metric(
                    "Predicted Temperature",
                    f"{predicted_temperature:.2f} °C"
                )


            with result_col2:

                st.metric(
                    "Features Used",
                    features_used
                )


            st.info(
                f"Model used: **{model_name}**"
            )


            # ------------------------------------------------
            # API RESPONSE
            # ------------------------------------------------

            with st.expander(
                "View API Response"
            ):

                st.json(result)


        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        else:

            st.error(
                f"Prediction failed. "
                f"HTTP Status: {response.status_code}"
            )

            try:

                st.json(
                    response.json()
                )

            except Exception:

                st.write(
                    response.text
                )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI."
        )

        st.warning(
            "Start the FastAPI server first using:\n\n"
            "python -m uvicorn api.app:app "
            "--reload --port 8001"
        )


    except requests.exceptions.Timeout:

        st.error(
            "❌ FastAPI request timed out."
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Weather Intelligence AI • FastAPI + Streamlit • "
    "RandomForestRegressor"
)