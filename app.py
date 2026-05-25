from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""
    news = ""

    if request.method == "POST":

        news = request.form["news"]

        # Transform text
        news_vector = vectorizer.transform([news])

        # Predict
        result = model.predict(news_vector)

        # Confidence score
        probability = model.predict_proba(news_vector)

        confidence = round(max(probability[0]) * 100, 2)

        if result[0] == 0:
            prediction = "Fake News"
        else:
            prediction = "Real News"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news=news
    )

if __name__ == "__main__":
    app.run(debug=True)