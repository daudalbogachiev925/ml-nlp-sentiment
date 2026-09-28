"""Анализ тональности текста."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# === Данные ===
data = {
    "text": [
        "Отличный продукт, всем рекомендую",
        "Ужасное качество, верните деньги",
        "Всё понравилось, спасибо",
        "Не работает, полный отстой",
        "Хорошо, но могло быть лучше",
        "Отвратительный сервис",
        "Быстрая доставка, доволен",
        "Обман, не покупайте",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# === Пайплайн ===
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# === Инференс ===
new_texts = ["Прекрасное качество", "Не рекомендую, плохо"]
predictions = pipeline.predict(new_texts)
for text, pred in zip(new_texts, predictions):
    print(f"{text} → {'позитив' if pred == 1 else 'негатив'}")
