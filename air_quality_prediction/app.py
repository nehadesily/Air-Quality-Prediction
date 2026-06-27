from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("air_quality_model.pkl")
except Exception as e:
    model = None
    print("Error loading model:", e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Air Quality Prediction API is running!",
        "available_endpoints": {
            "GET /": "API Status",
            "GET /predict": "How to use the prediction endpoint",
            "POST /predict": "Predict AQI"
        }
    })


# GET request for /predict
@app.route("/predict", methods=["GET"])
def predict_get():
    return jsonify({
        "message": "Use POST method to predict air quality.",
        "example_request": {
            "SOi": 10,
            "Noi": 20,
            "Rpi": 30,
            "SPMi": 40
        }
    })


# POST request for /predict
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model could not be loaded."
        }), 500

    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Request body must be valid JSON."
            }), 400

        required_features = [
            "SOi",
            "Noi",
            "Rpi",
            "SPMi"
        ]

        missing = [feature for feature in required_features if feature not in data]

        if missing:
            return jsonify({
                "error": "Missing features",
                "missing": missing
            }), 400

        features = np.array([[
            float(data["SOi"]),
            float(data["Noi"]),
            float(data["Rpi"]),
            float(data["SPMi"])
        ]])

        prediction = model.predict(features)

        return jsonify({
            "input": {
                "SOi": data["SOi"],
                "Noi": data["Noi"],
                "Rpi": data["Rpi"],
                "SPMi": data["SPMi"]
            },
            "prediction": str(prediction[0])
        })

    except ValueError:
        return jsonify({
            "error": "All input values must be numeric."
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)