import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import resample

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

# Load data
train_data = pd.read_csv(r'D:\1. PRAZZU\My Projects\Movie Genres\archive\Genre Classification Dataset\train_data.txt', sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv(r'D:\1. PRAZZU\My Projects\Movie Genres\archive\Genre Classification Dataset\test_data.txt', sep=':::', engine='python', names=['ID', 'TITLE', 'DESCRIPTION'])
test_solution = pd.read_csv(r'D:\1. PRAZZU\My Projects\Movie Genres\archive\Genre Classification Dataset\test_data_solution.txt', sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

# Strip whitespace
train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()
test_solution.columns = test_solution.columns.str.strip()
train_data = train_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
test_data = test_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
test_solution = test_solution.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Downsample common genres (optional, uncomment to use)
"""
genres = train_data['GENRE'].unique()
balanced_data = []
for genre in genres:
    genre_data = train_data[train_data['GENRE'] == genre]
    if len(genre_data) > 2000:
        genre_data = resample(genre_data, replace=False, n_samples=2000, random_state=42)
    balanced_data.append(genre_data)
train_data = pd.concat(balanced_data)
print("New Genre Counts:")
print(train_data['GENRE'].value_counts())
"""

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
train_data['clean_description'] = train_data['DESCRIPTION'].apply(clean_text)
test_data['clean_description'] = test_data['DESCRIPTION'].apply(clean_text)

# Convert to TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data['clean_description'])
y_train = train_data['GENRE']
X_test = vectorizer.transform(test_data['clean_description'])

# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Train Logistic Regression with class weights
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_test = test_solution['GENRE']
print("Logistic Regression Accuracy with Class Weights:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Visualize genres
genre_counts = train_data['GENRE'].value_counts()
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar', color=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#36A2EB', '#FFCE56'])
plt.title('Movie Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Save model and vectorizer
joblib.dump(model, 'movie_genre_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Saved model as 'movie_genre_model.pkl' and vectorizer as 'tfidf_vectorizer.pkl'")

# Example predictions
new_plots = [
    "A spy races to stop a villain from destroying the city.",
    "A family goes on a magical adventure.",
    "A group of friends solve a murder mystery.",
    "A soldier fights in a historical battle.",
    "A scientist builds a time machine."
]
for plot in new_plots:
    new_plot_clean = [clean_text(plot)]
    new_plot_tfidf = vectorizer.transform(new_plot_clean)
    prediction = model.predict(new_plot_tfidf)
    print(f"Predicted Genre for '{plot}':", prediction[0])