import requests


# FastAPI prediction endpoint
API_URL = "http://127.0.0.1:8001/predict"


def predict_temperature(values):
    """
    Weather prediction tool.

    Sends 13 weather input parameters to the
    FastAPI prediction endpoint.
    """

    # Validate number of inputs
    if len(values) != 13:
        return {
            "success": False,
            "error": f"Expected 13 values, but received {len(values)}"
        }

    try:
        response = requests.post(
            API_URL,
            json={"values": values},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            return {
                "success": True,
                "prediction": result.get("prediction"),
                "unit": result.get("unit", "°C"),
                "model": result.get("model", "Weather AI Model")
            }

        return {
            "success": False,
            "error": f"API returned status code {response.status_code}",
            "details": response.text
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Weather FastAPI server is not running."
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Weather prediction API request timed out."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":

    print("=" * 70)
    print("WEATHER PREDICTION TOOL TEST")
    print("=" * 70)

    # 13 parameters required by the Weather Prediction API
    test_values = [
        20,      # 1. Summary
        0,       # 2. Precipitation Type
        18,      # 3. Apparent Temperature
        0.65,    # 4. Humidity
        14,      # 5. Wind Speed
        225,     # 6. Wind Bearing
        12,      # 7. Visibility
        2,       # 8. Cloud Cover
        1008,    # 9. Atmospheric Pressure
        150,     # 10. Daily Summary
        2026,    # 11. Year
        9,       # 12. Month
        11       # 13. Day
    ]

    print("\nSending 13 weather parameters to FastAPI...")

    result = predict_temperature(test_values)

    print("\nPrediction Result:")
    print(result)

    print()
    print("=" * 70)
    print("WEATHER TOOL TEST COMPLETED")
    print("=" * 70)