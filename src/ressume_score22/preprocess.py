# src/preprocess.py

import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('punkt', download_dir=r"E:\nltk")
nltk.download('punkt_tab', download_dir=r"E:\nltk")
nltk.download('stopwords', download_dir=r"E:\nltk")
nltk.download('wordnet', download_dir=r"E:\nltk")


# Global objects (efficient)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess a single text document
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def get_most_common_words(df, text_column: str, top_n: int = 20):
    """
    Get most common words from a dataframe column
    """
    all_text = ' '.join(df[text_column].astype(str))

    # Simple cleaning
    all_text = re.sub(r'\W', ' ', all_text.lower())

    tokens = word_tokenize(all_text)

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    common_words = Counter(filtered_tokens).most_common(top_n)

    return common_words


# ---------- Test Run ----------
if __name__ == "__main__":
    import pandas as pd

    # Example CSV path
    csv_path = r"E:\Ressumme_nlp\archive\Resume\Resume.csv"

    df = pd.read_csv(csv_path)

    print("Sample Preprocessed Resume:\n")
    print(preprocess_text(df['Resume_str'].iloc[0]))

    print("\nTop 20 Common Words Overall:\n")
    print(get_most_common_words(df, "Resume_str"))
