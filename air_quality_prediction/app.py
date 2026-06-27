from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("air_quality_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Air Quality Prediction API is running.",
        "endpoints": {
            "POST /predict": "Predict AQI Range"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        required_features = [
            "SOi",
            "Noi",
            "Rpi",
            "SPMi"
        ]

        # Check missing values
        missing = [f for f in required_features if f not in data]

        if missing:
            return jsonify({
                "error": f"Missing features: {missing}"
            }), 400

        features = np.array([
            [
                float(data["SOi"]),
                float(data["Noi"]),
                float(data["Rpi"]),
                float(data["SPMi"])
            ]
        ])

        prediction = model.predict(features)

        return jsonify({
            "prediction": str(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)