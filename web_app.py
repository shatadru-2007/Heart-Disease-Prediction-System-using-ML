from flask import Flask, render_template, request
import numpy as nmp
import joblib

app = Flask(__name__) 

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        data = nmp.array([data])
        data = scaler.transform(data)

        result = model.predict(data)

        if result[0] == 1:
            prediction = "Heart Disease Detected"
        else:
            prediction = "No Heart Disease"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)