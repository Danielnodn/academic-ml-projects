from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("simple_model.pkl", "rb"))

# Class labels
labels = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


@app.route("/")
def home():
    return {"message": "Iris Prediction API running"}


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": labels[prediction]
    })


if __name__ == "__main__":
    app.run(debug=True)