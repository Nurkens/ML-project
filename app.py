from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка модели и scaler
model = joblib.load("heart_disease_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Получение значений из формы
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        # Преобразуем в массив и масштабируем
        data = np.array([features])
        data_scaled = scaler.transform(data)

        # Предсказание
        prediction = model.predict(data_scaled)[0]

        result = "✅ Риск сердечного заболевания обнаружен" if prediction == 1 else "✅ Риск сердечного заболевания не обнаружен"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"⚠️ Ошибка: {e}")

if __name__ == "__main__":
    app.run(debug=True)