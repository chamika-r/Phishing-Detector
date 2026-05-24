from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model when the server starts
print("Loading model...")
model = joblib.load('models/phishing_model.pkl')
feature_columns = None  # loaded from the data

# Load feature column names from the dataset
df = pd.read_csv('data/emails.csv')
feature_columns = df.drop(columns=['Email No.', 'Prediction']).columns.tolist()
print(f"Model loaded. Expecting {len(feature_columns)} features.")

@app.route('/')
def home():
    return jsonify({
        "message": "Phishing Detector API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the email text from the request
    data = request.get_json()

    if not data or 'email_text' not in data:
        return jsonify({"error": "Please provide email_text"}), 400

    email_text = data['email_text'].lower()

    # Count word frequencies in the email
    word_counts = {}
    for word in feature_columns:
        word_counts[word] = email_text.split().count(word)

    # Convert to dataframe for the model
    input_df = pd.DataFrame([word_counts])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = {
        "prediction": "phishing" if prediction == 1 else "safe",
        "confidence": round(float(max(probability)) * 100, 2),
        "phishing_probability": round(float(probability[1]) * 100, 2),
        "safe_probability": round(float(probability[0]) * 100, 2)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)