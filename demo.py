import subprocess
import sys
import os
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Check and install required libraries
def install_requirements():
    try:
        import pandas
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier
        import nltk
    except ImportError as e:
        print(f"Error: {e}")
        print("Some required libraries are missing. Installing them now...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', 'scikit-learn', 'nltk', 'pandas'])
        print("Libraries installed successfully. PLEASE RESTART THE SCRIPT!")
        sys.exit()

# Call the function to check and install libraries
install_requirements()

# Download NLTK resources (only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function (same as before)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Lemmatization (reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Load the pre-trained model and vectorizer
def load_model_and_vectorizer():
    if os.path.exists('Model3.pkl') and os.path.exists('vectorizer3.pkl'):
        print("Loading pre-trained model and vectorizer...")
        with open('Model3.pkl', 'rb') as f:
            best_model3 = pickle.load(f)
        with open('vectorizer3.pkl', 'rb') as f:
            vectorizer3 = pickle.load(f)
        return best_model3, vectorizer3
    else:
        print("Model and vectorizer not found! Please train and save the model first.")
        sys.exit()

# Step 1: Load the pre-trained model and vectorizer
best_model3, vectorizer3 = load_model_and_vectorizer()

# Step 2: Read your custom review from a text file with UTF-8 encoding
def get_review_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            review = file.read().strip()
        return review
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please make sure the file exists.")
        sys.exit()
    except UnicodeDecodeError:
        print(f"Error: The file '{file_path}' contains invalid characters for UTF-8 decoding.")
        sys.exit()

# Step 3: Preprocess and vectorize the review
def preprocess_and_predict(review):
    # Preprocess the review text
    preprocessed_review = preprocess_text(review)
    
    # Vectorize the review using the pre-trained vectorizer
    review_vectorized = vectorizer3.transform([preprocessed_review])
    
    # Predict the rating using the pre-trained model
    prediction = best_model3.predict(review_vectorized)
    
    return prediction[0]  # Return the predicted rating

# Step 4: Main function to process the review and print results
def main():
    # Ask for the name of the text file from the user
    file_path = input("Please enter the path to the review text file (e.g., 'your_review.txt'): ").strip()
    
    # Read the review from the file
    review = get_review_from_file(file_path)
    
    # Extract the actual rating (last number in the review string, assuming the format you mentioned)
    actual_rating = int(review.split(',')[-1].strip())
    
    # Predict the rating
    predicted_rating = preprocess_and_predict(review)
    
    # Print the predicted vs actual rating
    print(f"\nReview: {review}")
    print(f"Predicted Rating: {predicted_rating}")
    print(f"Actual Rating: {actual_rating}")
    
    # Compare the predicted rating with the actual rating
    if predicted_rating == actual_rating:
        print("\nPrediction is correct!")
    else:
        print("\nPrediction is incorrect!")

if __name__ == "__main__":
    main()
