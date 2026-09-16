
/* ============================================================
   WEATHER INTELLIGENCE AI
   Frontend Prediction Engine
   ============================================================ */


/* ============================================================
   API CONFIGURATION
   ============================================================ */

const API_BASE_URL = window.location.origin;

const PREDICTION_ENDPOINT =
    `${API_BASE_URL}/predict`;

const MODEL_INFO_ENDPOINT =
    `${API_BASE_URL}/model-info`;


/* ============================================================
   HELPER — GET ELEMENT
   ============================================================ */

function getElement(id) {

    return document.getElementById(id);

}


/* ============================================================
   HELPER — GET NUMERIC VALUE
   ============================================================ */

function getNumericValue(id, fieldName) {

    const element = getElement(id);

    if (!element) {

        throw new Error(
            `${fieldName} input field is missing.`
        );

    }

    if (element.value.trim() === "") {

        throw new Error(
            `${fieldName} is required.`
        );

    }

    const value = Number(element.value);

    if (!Number.isFinite(value)) {

        throw new Error(
            `${fieldName} must contain a valid number.`
        );

    }

    return value;

}


/* ============================================================
   VALIDATE WEATHER INPUT
   ============================================================ */

function validateWeatherInput(data) {

    if (
        data.humidity < 0 ||
        data.humidity > 1
    ) {

        throw new Error(
            "Humidity must be between 0 and 1."
        );

    }


    if (data.wind_speed < 0) {

        throw new Error(
            "Wind speed cannot be negative."
        );

    }


    if (
        data.wind_bearing < 0 ||
        data.wind_bearing > 360
    ) {

        throw new Error(
            "Wind bearing must be between 0 and 360 degrees."
        );

    }


    if (data.visibility < 0) {

        throw new Error(
            "Visibility cannot be negative."
        );

    }


    if (data.pressure < 0) {

        throw new Error(
            "Pressure cannot be negative."
        );

    }


    if (
        data.year < 1900 ||
        data.year > 2100
    ) {

        throw new Error(
            "Please enter a valid year between 1900 and 2100."
        );

    }


    if (
        data.month < 1 ||
        data.month > 12
    ) {

        throw new Error(
            "Month must be between 1 and 12."
        );

    }


    if (
        data.day < 1 ||
        data.day > 31
    ) {

        throw new Error(
            "Day must be between 1 and 31."
        );

    }

}


/* ============================================================
   DYNAMIC MODEL INFORMATION
   ============================================================ */

async function loadModelInformation() {

    try {

        const response =
            await fetch(
                MODEL_INFO_ENDPOINT
            );

        if (!response.ok) {

            throw new Error(
                "Unable to load model information."
            );

        }

        const data =
            await response.json();


        /* ----------------------------------------------------
           CHECK MODEL STATUS
           ---------------------------------------------------- */

        if (!data.model_loaded) {

            console.warn(
                "AI model is not currently loaded."
            );

            return;

        }


        /* ----------------------------------------------------
           MODEL NAME
           ---------------------------------------------------- */

        const modelName =
            getElement("modelName");

        const aiModelName =
            getElement("aiModelName");


        if (modelName) {

            modelName.textContent =
                formatModelName(
                    data.model_name ||
                    "RandomForestRegressor"
                );

        }


        if (aiModelName) {

            aiModelName.textContent =
                formatModelName(
                    data.model_name ||
                    "RandomForestRegressor"
                );

        }


        /* ----------------------------------------------------
           FEATURE COUNT
           ---------------------------------------------------- */

        const featureCount =
            Number(
                data.feature_count
            );


        if (
            Number.isFinite(featureCount)
        ) {

            const parameterCount =
                getElement(
                    "parameterCount"
                );

            const inputParameterCount =
                getElement(
                    "inputParameterCount"
                );


            if (parameterCount) {

                parameterCount.textContent =
                    featureCount;

            }


            if (inputParameterCount) {

                inputParameterCount.textContent =
                    `${featureCount} Parameters`;

            }

        }


        /* ----------------------------------------------------
           CONSOLE INFORMATION
           ---------------------------------------------------- */

        console.log(
            "Dynamic model information loaded:",
            data
        );


    } catch (error) {

        console.error(
            "Model information could not be loaded:",
            error
        );

        /*
         * We intentionally do not stop the prediction engine.
         * Prediction can still work if /model-info fails.
         */

    }

}


/* ============================================================
   FORMAT MODEL NAME
   ============================================================ */

function formatModelName(modelName) {

    if (!modelName) {

        return "Random Forest";

    }


    return String(modelName)
        .replace(
            /Regressor$/i,
            ""
        )
        .replace(
            /([a-z])([A-Z])/g,
            "$1 $2"
        )
        .trim();

}


/* ============================================================
   SET LOADING STATE
   ============================================================ */

function setLoadingState(isLoading) {

    const loading =
        getElement("loading");

    const button =
        getElement("predictButton");


    if (loading) {

        loading.style.display =
            isLoading ? "flex" : "none";

    }


    if (button) {

        button.disabled =
            isLoading;


        if (isLoading) {

            button.dataset.originalText =
                button.innerHTML;

            button.innerHTML = `
                <span class="button-icon">⏳</span>
                <span>Analyzing Weather...</span>
            `;

        } else {

            button.innerHTML =
                button.dataset.originalText ||
                `
                    <span class="button-icon">🔮</span>
                    <span>Predict Temperature</span>
                    <span class="button-arrow">→</span>
                `;

        }

    }

}


/* ============================================================
   SHOW RESULT
   ============================================================ */

function showPrediction(
    temperatureValue
) {

    const result =
        getElement("result");

    const temperature =
        getElement("temperature");


    if (!result || !temperature) {

        throw new Error(
            "Prediction result elements are missing."
        );

    }


    const numericTemperature =
        Number(
            temperatureValue
        );


    if (
        !Number.isFinite(
            numericTemperature
        )
    ) {

        throw new Error(
            "The AI returned an invalid temperature value."
        );

    }


    temperature.textContent =
        `${numericTemperature.toFixed(2)} °C`;


    result.style.display =
        "flex";


    setTimeout(() => {

        result.scrollIntoView({

            behavior: "smooth",

            block: "center"

        });

    }, 100);

}


/* ============================================================
   HIDE RESULT
   ============================================================ */

function hideResult() {

    const result =
        getElement("result");


    if (result) {

        result.style.display =
            "none";

    }

}


/* ============================================================
   FORMAT API ERROR
   ============================================================ */

function formatApiError(detail) {

    if (!detail) {

        return "Prediction request failed.";

    }


    if (
        typeof detail === "string"
    ) {

        return detail;

    }


    if (
        typeof detail === "object"
    ) {

        if (detail.message) {

            return detail.message;

        }


        try {

            return JSON.stringify(
                detail,
                null,
                2
            );

        } catch {

            return (
                "An unexpected API error occurred."
            );

        }

    }


    return String(detail);

}


/* ============================================================
   MAIN PREDICTION FUNCTION
   ============================================================ */

async function predictTemperature() {

    hideResult();


    try {

        /* ----------------------------------------------------
           COLLECT ALL 13 MODEL PARAMETERS
           ---------------------------------------------------- */

        const data = {

            summary:
                getNumericValue(
                    "summary",
                    "Summary"
                ),

            precip_type:
                getNumericValue(
                    "precip_type",
                    "Precipitation type"
                ),

            apparent_temperature:
                getNumericValue(
                    "apparent_temperature",
                    "Apparent temperature"
                ),

            humidity:
                getNumericValue(
                    "humidity",
                    "Humidity"
                ),

            wind_speed:
                getNumericValue(
                    "wind_speed",
                    "Wind speed"
                ),

            wind_bearing:
                getNumericValue(
                    "wind_bearing",
                    "Wind bearing"
                ),

            visibility:
                getNumericValue(
                    "visibility",
                    "Visibility"
                ),

            cloud_cover:
                getNumericValue(
                    "cloud_cover",
                    "Cloud cover"
                ),

            pressure:
                getNumericValue(
                    "pressure",
                    "Pressure"
                ),

            daily_summary:
                getNumericValue(
                    "daily_summary",
                    "Daily summary"
                ),

            year:
                getNumericValue(
                    "year",
                    "Year"
                ),

            month:
                getNumericValue(
                    "month",
                    "Month"
                ),

            day:
                getNumericValue(
                    "day",
                    "Day"
                )

        };


        /* ----------------------------------------------------
           CLIENT-SIDE VALIDATION
           ---------------------------------------------------- */

        validateWeatherInput(
            data
        );


        /* ----------------------------------------------------
           SHOW LOADING STATE
           ---------------------------------------------------- */

        setLoadingState(
            true
        );


        /* ----------------------------------------------------
           SEND REQUEST TO FASTAPI
           ---------------------------------------------------- */

        const response =
            await fetch(
                PREDICTION_ENDPOINT,
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"

                    },

                    body:
                        JSON.stringify(
                            data
                        )

                }
            );


        /* ----------------------------------------------------
           READ API RESPONSE
           ---------------------------------------------------- */

        let resultData;


        try {

            resultData =
                await response.json();

        } catch {

            throw new Error(
                "The AI server returned an invalid response."
            );

        }


        /* ----------------------------------------------------
           HANDLE API ERRORS
           ---------------------------------------------------- */

        if (!response.ok) {

            throw new Error(
                formatApiError(
                    resultData.detail
                )
            );

        }

        /* ----------------------------------------------------
           GET PREDICTED TEMPERATURE
           ---------------------------------------------------- */

        let predictedTemperature;


        if (
            resultData.prediction &&
            typeof resultData
                .prediction
                .predicted_temperature_celsius
                !== "undefined"
        ) {

            predictedTemperature =
                resultData
                    .prediction
                    .predicted_temperature_celsius;

        }


        else if (
            typeof resultData
                .predicted_temperature_celsius
                !== "undefined"
        ) {

            predictedTemperature =
                resultData
                    .predicted_temperature_celsius;

        }


        else {

            throw new Error(
                "The AI server did not return a temperature prediction."
            );

        }


        /* ----------------------------------------------------
           DISPLAY RESULT
           ---------------------------------------------------- */

        showPrediction(
            predictedTemperature
        );


        console.log(
            "Weather prediction successful:",
            predictedTemperature
        );


    } catch (error) {

        console.error(
            "Weather Intelligence AI Error:",
            error
        );


        alert(
            "Weather Intelligence AI\n\n" +
            "Prediction could not be completed.\n\n" +
            "Reason: " +
            error.message
        );


    } finally {

        setLoadingState(
            false
        );

    }

}


/* ============================================================
   ENTER KEY SUPPORT
   ============================================================ */

document.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter" &&
            event.target.tagName === "INPUT"
        ) {

            event.preventDefault();

            predictTemperature();

        }

    }
);


/* ============================================================
   INITIALIZATION
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "Weather Intelligence AI frontend initialized."
        );

        console.log(
            "Prediction endpoint:",
            PREDICTION_ENDPOINT
        );


        /*
         * Load backend model information
         * when the prediction page opens.
         */

        loadModelInformation();

    }
);

