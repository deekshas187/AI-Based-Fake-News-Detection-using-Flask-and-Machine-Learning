from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def is_valid_news(text):

    if not text or text.strip() == "":
        return "Please enter news text!"

    text = text.strip()

    if len(text) < 20:
        return "Input too short to analyze!"

    # reject random symbols / garbage input
    if re.match(r"^[^a-zA-Z]+$", text):
        return "Invalid news format!"

    return None


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""
    news = ""
    error = ""

    if request.method == "POST":

        news = request.form["news"]

        # CHECK INPUT FIRST
        error = is_valid_news(news)

        if error:
            return render_template(
                "index.html",
                prediction=error,
                confidence="",
                news=news
            )

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
