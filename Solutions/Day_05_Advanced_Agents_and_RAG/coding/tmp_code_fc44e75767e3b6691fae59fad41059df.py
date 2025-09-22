from flask import Flask, request, jsonify
import re

app = Flask(__name__)

def calculate_password_complexity(password):
    length_score = 0
    char_type_score = {
        "uppercase": 0,
        "lowercase": 0,
        "numbers": 0,
        "symbols": 0
    }

    length_of_password = len(password)

    # Check length criteria
    if length_of_password < 8:
        length_score = 1
    elif 8 <= length_of_password <= 12:
        length_score = 2
    else:
        length_score = 3

    # Character type criteria
    if re.search(r"[A-Z]", password):
        char_type_score["uppercase"] = 1
    if re.search(r"[a-z]", password):
        char_type_score["lowercase"] = 1
    if re.search(r"[0-9]", password):
        char_type_score["numbers"] = 1
    if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        char_type_score["symbols"] = 1

    total_score = length_score + sum(char_type_score.values())

    # Determine complexity level
    if total_score <= 2:
        complexity_level = "Weak"
    elif total_score == 3 or total_score == 4:
        complexity_level = "Moderate"
    elif total_score == 5:
        complexity_level = "Strong"
    else:
        complexity_level = "Very Strong"

    return total_score, complexity_level

@app.route('/api/password-complexity', methods=['POST'])
def password_complexity():
    data = request.json
    password = data.get('password', '')

    if not password:
        return jsonify({"error": "Password is required"}), 400

    complexity_score, complexity_level = calculate_password_complexity(password)

    return jsonify({
        "complexityScore": complexity_score,
        "complexityLevel": complexity_level
    })

if __name__ == '__main__':
    app.run(debug=True)