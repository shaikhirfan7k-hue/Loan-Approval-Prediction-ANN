
"""Flask GUI for the trained loan approval ANN."""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("loan_ann_model.keras")
preprocessor = joblib.load("loan_preprocessor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-form")
def predict_form():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        applicant = {
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
            "Credit_History": float(request.form["Credit_History"]),
            "Property_Area": request.form["Property_Area"],
        }

        # Convert applicant details to a 2D DataFrame
        applicant_df = pd.DataFrame([applicant])

        # Preprocess applicant data
        data = preprocessor.transform(applicant_df)

        if hasattr(data, "toarray"):
            data = data.toarray()

        # Make ANN prediction
        probability = float(model.predict(data, verbose=0)[0][0])
        approved = probability >= 0.5

        return render_template(
            "result.html",
            approved=approved,
            probability=probability,
            applicant=applicant
        )

    except Exception as err:
        return render_template(
            "predict.html",
            error=str(err)
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
