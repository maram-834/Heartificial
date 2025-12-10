from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Load your tuned Random Forest model
model = joblib.load("heart_disease_rf_model.pkl")

# Simple user storage (in production, use a real database)
USERS_FILE = '/tmp/users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

@app.route('/')
def login_page():
    if 'user' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    users = load_users()
    
    email = data.get('email')
    password = data.get('password')
    
    if email in users and check_password_hash(users[email]['password'], password):
        session['user'] = email
        session['name'] = users[email]['name']
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    users = load_users()
    
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    
    if email in users:
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    users[email] = {
        'name': name,
        'password': generate_password_hash(password)
    }
    save_users(users)
    
    session['user'] = email
    session['name'] = name
    return jsonify({'success': True})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()

    feature_order = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    try:
        features = np.array([float(data[f]) for f in feature_order]).reshape(1, -1)

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        no_disease = round(prob[0] * 100, 2)
        disease = round(prob[1] * 100, 2)

        # Custom messages
        if pred == 1:
            risk_level = "High Risk"
            message = "⚠️ High risk detected. Please consult a cardiologist immediately for a comprehensive evaluation."
        else:
            risk_level = "Low Risk"
            message = "✅ Low risk detected. Maintain a healthy lifestyle and regular check-ups."

        result = {
            "risk_level": risk_level,
            "probability": {
                "no_disease": no_disease,
                "disease": disease
            },
            "message": message,
            "input_data": data,
            "user_name": session.get('name', 'N/A'),
            "user_email": session.get('user', 'N/A'),
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
