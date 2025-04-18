import sqlite3
import json

conn = sqlite3.connect("feedback/feedback.db")
cursor = conn.cursor()
cursor.execute("SELECT query, response, feedback FROM feedback_log WHERE feedback = 'up'")
data = cursor.fetchall()
conn.close()

formatted = []
for q, a, _ in data:
    formatted.append({
        "question": q,
        "context": "",  # Optional: You can store original context if needed
        "answer": a
    })

with open("training/fine_tune_data.json", "w") as f:
    json.dump(formatted, f, indent=2)