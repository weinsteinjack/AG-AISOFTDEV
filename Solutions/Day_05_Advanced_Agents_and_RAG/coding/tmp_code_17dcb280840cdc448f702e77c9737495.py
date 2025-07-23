from flask import Flask, request, jsonify
import re

app = Flask(__name__)

def calculate_password_complexity(password):
    length_score = 0
    complexity_score = 0
    min_length = 8

    # Score based on length
    if len(password) >= min_length:
        length_score = 10 + (len(password) - min_length) * 2
    else:
        length_score = len(password) * 1.25

    complexity_score += length_score

    # Check for uppercase letters
    if re.search(r'[A-Z]', password):
        complexity_score += 10

    # Check for lowercase letters
    if re.search(r'[a-z]', password):
        complexity_score += 10

    # Check for digits
    if re.search(r'[0-9]', password):
        complexity_score += 10

    # Check for symbols
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        complexity_score += 10

    # Categorization based on score
    if complexity_score >= 81:
        category = "Very Strong"
    elif 61 <= complexity_score < 81:
        category = "Strong"
    elif 41 <= complexity_score < 61:
        category = "Moderate"
    elif 21 <= complexity_score < 41:
        category = "Weak"
    else:
        category = "Very Weak"
    
    return complexity_score, category

@app.route('/api/v1/password-complexity', methods=['POST'])
def password_complexity():
    data = request.get_json()
    if 'password' not in data:
        return jsonify({"error": "Password not provided"}), 400

    password = data['password']
    complexity_score, category = calculate_password_complexity(password)
    
    return jsonify({"complexity_score": complexity_score, "category": category})

if __name__ == '__main__':
    app.run(debug=True)