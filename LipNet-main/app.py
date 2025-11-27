from flask import Flask, request, jsonify
import psycopg2
from psycopg2 import sql, errors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ----------------- DATABASE CONFIG -----------------
db_config = {
    "host": "localhost",
    "database": "lipbuddy",
    "user": "lipuser",
    "password": "meghana123",
    "port": 5432  # default for PostgreSQL
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        print("❌ Database connection error:", e)
        return None

# ----------------- SIGNUP -----------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"status": "error", "message": "Cannot connect to database"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, password)
        )
        conn.commit()
        cursor.close()
        return jsonify({"status": "success", "message": "User registered successfully"}), 201

    except errors.UniqueViolation:
        conn.rollback()
        return jsonify({"status": "error", "message": "Username already exists"}), 409

    except Exception as err:
        conn.rollback()
        return jsonify({"status": "error", "message": str(err)}), 500

    finally:
        conn.close()

# ----------------- LOGIN -----------------
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"status": "error", "message": "Cannot connect to database"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password = %s",
            (username, password)
        )
        user = cursor.fetchone()
        cursor.close()

        if user:
            return jsonify({"status": "success", "message": "Login successful"}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    except Exception as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    finally:
        conn.close()

# ----------------- HEALTH CHECK -----------------
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"message": "Backend live ✅"}), 200

if __name__ == '__main__':
    app.run(port=5050, debug=True)