from flask import Flask, request, jsonify

app = Flask(__name__)

def calculate_password_complexity(password):
    length = len(password)
    has_uppercase = any(c.isupper() for c in password)
    has_lowercase = any(c.islower() for c in password)
    has_numbers = any(c.isdigit() for c in password)
    has_symbols = any(not c.isalnum() for c in password)

    # Scoring based on length
    if length >= 15:
        length_score = 3
    elif length >= 11:
        length_score = 2
    elif length >= 8:
        length_score = 1
    else:
        length_score = 0

    # Scoring based on character types
    type_score = sum([has_uppercase, has_lowercase, has_numbers, has_symbols])

    # Total score
    total_score = length_score + type_score

    # Determine complexity description
    if total_score <= 2:
        description = "Weak"
    elif total_score <= 4:
        description = "Moderate"
    else:
        description = "Strong"

    return total_score, description

@app.route('/api/password/complexity', methods=['POST'])
def password_complexity():
    data = request.get_json()

    if not data or 'password' not in data:
        return jsonify({"error": "Password is required"}), 400

    password = data['password']
    if not password:
        return jsonify({"error": "Password cannot be empty"}), 400
    
    score, description = calculate_password_complexity(password)
    return jsonify({"score": score, "description": description})

if __name__ == '__main__':
    app.run(debug=True)