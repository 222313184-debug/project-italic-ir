from flask import Flask, render_template, request
import numpy as np
import joblib

from stylometry import stylometry_31   # PENTING: pakai 31 fitur

app = Flask(__name__)

# === Load model terbaik (Random Forest) + scaler 31 fitur ===
# SESUAIKAN nama file .pkl dengan yang kamu simpan dari Colab
rf_model = joblib.load("models/rf_stylo31.pkl")          # contoh nama file
stylo_scaler = joblib.load("models/stylo31_scaler.pkl")  # scaler untuk 31 fitur

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    prob_ai = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("input_text", "")

        # 1. Hitung 31 fitur stylometri dari teks input
        feat = np.array(stylometry_31(input_text)).reshape(1, -1)

        # 2. Scaling (harus pakai scaler yang sama dengan waktu training)
        feat_scaled = stylo_scaler.transform(feat)

        # 3. Prediksi probabilitas kelas AI (kelas 1)
        proba = rf_model.predict_proba(feat_scaled)[0]
        prob_ai = float(proba[1])

        # 4. Prediksi kelas (threshold default 0.5 dari model)
        y_pred = rf_model.predict(feat_scaled)[0]

        # 5. Mapping label → kalimat
        if y_pred == 1:
            prediction_text = "Teks terdeteksi sebagai HASIL AI."
        else:
            prediction_text = "Teks terdeteksi sebagai TULISAN MANUSIA."

    return render_template(
        "index.html",
        prediction=prediction_text,
        prob_ai=prob_ai,
        input_text=input_text,
    )

if __name__ == "__main__":
    app.run(debug=True)
