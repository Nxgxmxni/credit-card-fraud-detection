from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
print("Template folder location:", app.template_folder)
# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model/fraud_model.pkl")
training_columns = model.feature_names_in_

# ----------------------------
# Setup Database
# ----------------------------
conn = sqlite3.connect("transactions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL,
    prediction TEXT,
    fraud_probability REAL
)
""")
conn.commit()

# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# Predict Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "amount": float(request.form["amount"]),
            "transaction_hour": int(request.form["transaction_hour"]),
            "foreign_transaction": int(request.form["foreign_transaction"]),
            "location_mismatch": int(request.form["location_mismatch"]),
            "device_trust_score": float(request.form["device_trust_score"]),
            "velocity_last_24h": int(request.form["velocity_last_24h"]),
            "cardholder_age": int(request.form["cardholder_age"]),
            "merchant_category": request.form["merchant_category"]
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)
        df = df.reindex(columns=training_columns, fill_value=0)

        probability = model.predict_proba(df)[0][1]
        threshold = 0.55

        result = "Fraud" if probability >= threshold else "Legitimate"
        fraud_percentage = float(round(probability * 100, 2))

        # Risk Level System
        if fraud_percentage < 40:
            risk = "Low"
            color = "green"
        elif fraud_percentage < 70:
            risk = "Medium"
            color = "orange"
        else:
            risk = "High"
            color = "red"

        cursor.execute(
            "INSERT INTO transactions (amount, prediction, fraud_probability) VALUES (?, ?, ?)",
            (data["amount"], result, fraud_percentage)
        )
        conn.commit()

        return render_template(
            "index.html",
            prediction_text=result,
            fraud_prob=fraud_percentage,
            risk=risk,
            color=color
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ----------------------------
# Bulk Upload
# ----------------------------
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    if not file or file.filename == "":
        return "No file selected."

    filename = file.filename.lower()
    df = None

    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(file)
        elif filename.endswith(".csv"):
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin1")
        else:
            return "Unsupported file format. Please upload CSV or XLSX."
    except Exception as e:
        return f"File reading error: {str(e)}"

    if df is None or df.empty:
        return "Uploaded file is empty."

    df = pd.get_dummies(df)
    df = df.reindex(columns=training_columns, fill_value=0)

    probabilities = model.predict_proba(df)[:, 1]

    df["Fraud_Probability (%)"] = (probabilities * 100).round(2)
    df["Prediction"] = ["Fraud" if p >= 0.55 else "Legitimate" for p in probabilities]

    fraud_count = int(sum(df["Prediction"] == "Fraud"))
    total = len(df)

    os.makedirs("static", exist_ok=True)
    output_path = "static/bulk_results.xlsx"
    df.to_excel(output_path, index=False)

    return render_template(
        "bulk_result.html",
        table=df.head(20).to_html(index=False),
        fraud_count=fraud_count,
        total=total,
        download_link=output_path
    )

# ----------------------------
# Dashboard
# ----------------------------
@app.route("/dashboard")
def dashboard():

    cursor.execute("SELECT * FROM transactions")
    rows = cursor.fetchall()

    fraud_count = sum(1 for row in rows if row[2] == "Fraud")
    legit_count = len(rows) - fraud_count

    os.makedirs("static", exist_ok=True)

    plt.figure()
    plt.pie(
        [legit_count, fraud_count],
        labels=["Legitimate", "Fraud"],
        autopct="%1.1f%%"
    )
    plt.title("Fraud vs Legitimate Transactions")
    plt.savefig("static/dashboard_pie.png")
    plt.close()

    return render_template(
        "dashboard.html",
        rows=rows,
        fraud_count=fraud_count,
        legit_count=legit_count
    )

# ----------------------------
# Delete Transaction
# ----------------------------
@app.route("/delete/<int:id>")
def delete_transaction(id):
    cursor.execute("DELETE FROM transactions WHERE id = ?", (id,))
    conn.commit()
    return redirect(url_for("dashboard"))

# ----------------------------
# Clear All
# ----------------------------
@app.route("/clear")
def clear_all():
    cursor.execute("DELETE FROM transactions")
    conn.commit()
    return redirect(url_for("dashboard"))

# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)