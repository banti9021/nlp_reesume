# model.py

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# ----------------------------
# Text Preprocessing
# ----------------------------
def clean_text(text: str) -> str:
    """
    Clean and preprocess resume text:
    - Lowercase
    - Remove numbers & punctuation
    - Remove stopwords
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)


# ----------------------------
# Train TF-IDF + MultinomialNB Model
# ----------------------------
def train_resume_classifier(
    df: pd.DataFrame, 
    text_col: str = 'cleaned_resume', 
    target_col: str = 'Category', 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Train a TF-IDF + MultinomialNB text classifier on resume dataset.

    Returns:
        pipeline: trained sklearn pipeline
        X_test, y_test, y_pred: test data & predictions
        cm: confusion matrix
    """
    # Split data
    X = df[text_col]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Pipeline: TF-IDF + Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return pipeline, X_test, y_test, y_pred, cm


# ----------------------------
# Plot Confusion Matrix
# ----------------------------
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example: Load combined CSV using your data_loader
    from data_loader import load_csv_folder
    
    folder_path = r"E:\Ressumme_nlp\archive\Resume"
    df = load_csv_folder(folder_path)
    
    # Apply text cleaning
    df['cleaned_resume'] = df['Resume_Text'].apply(clean_text)
    
    # Train model
    pipeline, X_test, y_test, y_pred, cm = train_resume_classifier(df)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, labels=df['Category'].unique())
