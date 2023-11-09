from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


# Load the model and scaler
def getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS):
    try:
        # Scale the input data using the loaded scaler
        model = pickle.load(open("model/titanic_survival_ml_model.sav", "rb"))
        scaler = pickle.load(open("model/scaler.sav", "rb"))
        input_data = [[pclass, sex, age, sibsp, parch, fare, embC, embQ, embS]]
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        if prediction == 0:
            return "not survived"
        elif prediction == 1:
            return "survived"
        else:
            return "error"
    except Exception as e:
        return str(e)


# Home page route
@app.route("/")
def home():
    return render_template("index.html")


# Result page route
@app.route("/result", methods=["GET"])
def result():
    if request.method == "GET":
        pclass = int(
            request.args.get("pclass", "3")
        )  # Default to class 3 if not provided
        sex = int(request.args.get("sex", "0"))  # Default to male if not provided
        age = float(request.args.get("age", "30"))  # Default age to 30 if not provided
        sibsp = int(request.args.get("sibsp", "0"))  # Default to 0 if not provided
        parch = int(request.args.get("parch", "0"))  # Default to 0 if not provided
        fare = float(request.args.get("fare", "0.0"))  # Default to 0.0 if not provided
        embC = int(request.args.get("embC", "0"))
        embQ = int(request.args.get("embQ", "0"))
        embS = int(request.args.get("embS", "0"))

        result = getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS)

        return render_template("result.html", result=result)

    return "Invalid request method"


if __name__ == "__main__":
    app.run(debug=True)
