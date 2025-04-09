import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

texts = [
    "I hate this", "You are awful", "This is disgusting", "I can't stand you", "Get lost",
    "You suck", "You're stupid", "You're the worst",
    "I love this", "You are great", "Wonderful work", "Keep it up", "This is amazing",
    "Nice job", "That's kind of you", "Respectful and thoughtful", "Great effort", "You did well",
    "Shut up", "Idiot", "Fool", "You're horrible", "So mean","Nigga","Fuck You","Bitch"
]
labels = [1,1,1,1,1, 1,1,1, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1,1,1,1]


# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train the model
model = LinearSVC()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, "grid_search_svc.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
