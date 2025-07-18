from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from predict import predict_image_bytes
import wikipedia
import mysql.connector
import os
import datetime
import time

# ───── Setup Flask ─────
app = Flask(__name__)
CORS(app)

# ───── Upload Folder ─────
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE_MB = 5
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ───── MySQL Connection ─────
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='plantclassificationdb'
    )

# ───── Wikipedia Summary ─────
def get_wikipedia_summary(plant_name):
    try:
        query = f"{plant_name} plant"  # Tambah 'plant' untuk pastikan carian berkaitan tumbuhan
        print(f"🔍 Searching Wikipedia for: {query}")

        search_results = wikipedia.search(query)
        if not search_results:
            print("⚠️ No Wikipedia results found.")
            return "No plant-related description found."

        summary = wikipedia.summary(search_results[0], sentences=2)
        return summary[:500]

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Disambiguation warning: {e.options}")
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
            return summary[:500]
        except:
            return "No plant-related description found."
    except Exception as e:
        print(f"⚠️ Wikipedia error: {e}")
        return "No plant-related description found."

# ───── Health Check ─────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "message": "Plant Few-Shot API is running"
    }), 200

# ───── Predict Only ─────
@app.route("/predict", methods=["POST"])
def predict_route():
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        }), 400

    try:
        image_bytes = request.files['image'].read()
        start = time.time()
        predicted_class = predict_image_bytes(image_bytes)
        print(f"✅ Prediction done in {time.time() - start:.2f}s")

        summary_text = get_wikipedia_summary(predicted_class)

        return jsonify({
            "status": "success",
            "prediction": predicted_class,
            "description": summary_text
        }), 200

    except Exception as e:
        print(f"❌ Predict error: {e}")
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "details": str(e)
        }), 500

# ───── Predict and Store ─────
@app.route("/predict_and_store", methods=["POST"])
def predict_and_store():
    print("📥 Received POST /predict_and_store")
    image = request.files.get('image')
    user_id = request.form.get('user_id')

    if not image or not user_id:
        return jsonify({
            "status": "error",
            "message": "Missing image or user_id"
        }), 400

    # ─ Check image size ─
    image.seek(0, os.SEEK_END)
    if image.tell() / (1024 * 1024) > MAX_FILE_SIZE_MB:
        return jsonify({
            "status": "error",
            "message": f"Image too large (>{MAX_FILE_SIZE_MB}MB)"
        }), 400
    image.seek(0)

    # ─ Save image ─
    filename = secure_filename(f"{datetime.datetime.now():%Y%m%d%H%M%S}_{image.filename}")
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)
    print(f"📁 Image saved to: {image_path}")

    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        print("🧠 Predicting...")
        start_time = time.time()
        plant_name = predict_image_bytes(image_bytes)

        if not plant_name:
            raise Exception("Model returned no prediction")

        print(f"🌿 Plant: {plant_name} in {time.time() - start_time:.2f}s")
        plant_name = plant_name.strip()

        description = get_wikipedia_summary(plant_name)
        print("📚 Wikipedia summary fetched")

        # ─ Store to DB ─
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT user_id FROM tbl_users WHERE user_id = %s", (user_id,))
        if not cursor.fetchone():
            return jsonify({
                "status": "error",
                "message": f"user_id {user_id} not found"
            }), 400

        cursor.execute("""
            INSERT INTO predictions (user_id, plant_name, description)
            VALUES (%s, %s, %s)
        """, (user_id, plant_name, description))
        conn.commit()
        print("✅ Prediction stored in database")

        return jsonify({
            "status": "success",
            "plant_name": plant_name,
            "description": description
        }), 200

    except Exception as e:
        print(f"❌ Error in /predict_and_store: {e}")
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "details": str(e)
        }), 500

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals() and conn.is_connected(): conn.close()

# ───── Get Prediction History ─────
@app.route('/get_predictions/<int:user_id>', methods=['GET'])
def get_predictions(user_id):
    print(f"📥 GET /get_predictions/{user_id}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT plant_name, description, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        results = cursor.fetchall()
        print(f"📄 Retrieved {len(results)} predictions")

        return jsonify(results), 200

    except Exception as e:
        print(f"❌ History error: {e}")
        return jsonify({
            "status": "error",
            "message": "Database error",
            "details": str(e)
        }), 500

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals() and conn.is_connected(): conn.close()

# ───── Run Flask ─────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)