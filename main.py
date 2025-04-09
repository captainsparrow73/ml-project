
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("grid_search_svc.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# HTML template with a simple form
form_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Hate Speech Detector</title>
</head>
<body>
    <h2>🧠 Hate Speech Detection</h2>
    <form action="/predict_form" method="post">
        Enter Text:<br>
        <textarea name="text" rows="5" cols="60" placeholder="Type something..."></textarea><br><br>
        <input type="submit" value="Check">
    </form>

    {% if prediction is not none %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(form_template, prediction=None)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        text = request.form.get("text", "")
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        result = "🚫 Hate Speech Detected!" if prediction == 1 else "✅ Not Hate Speech."
        return render_template_string(form_template, prediction=result)
    except Exception as e:
        return f"<p>Error: {str(e)}</p>"

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



