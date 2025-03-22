import subprocess
import sys

# Step 1: Check and install required libraries
def install_requirements():
    try:
        # Try to import all required libraries
        import pandas
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import nltk
        from imblearn.over_sampling import SMOTE
    except ImportError:
        # If any library is missing, install using requirements.txt
        print("Some required libraries are missing. Installing them now...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Libraries installed successfully. PLEASE RESTART THE SCRIPT!")
        sys.exit()

# Call the function to check and install libraries
install_requirements()

# Step 2: Import libraries (now they should be installed)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Step 3: Download NLTK resources (only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Step 4: Load the data
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Step 5: Map ratings from [1, 2, 3, 4, 5] to [0, 1, 2, 3, 4]
y = df['Rating'] - 1

# Step 6: Advanced text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Lemmatization (reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['Review'] = df['Review'].apply(preprocess_text)

# Step 7: Feature extraction with n-grams
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  # Adjusted values
X = vectorizer.fit_transform(df['Review'])

# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 10: Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [3, 5, 7],      # Maximum depth of trees
    'learning_rate': [0.01, 0.1],  # Learning rate
    'subsample': [0.8, 1.0]       # Subsample ratio
}

model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# Best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Step 11: Prediction
y_pred = best_model.predict(X_test)

# Step 12: Save predictions to a text file
with open('prediction.txt', 'w') as file:
    for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
        correct = "Correct" if pred == actual else "Incorrect"
        file.write(f"{i + 1}: Prediction \"{pred}\", Actual Rating \"{actual + 1}\", {correct}\n")  # Map back to [1, 2, 3, 4, 5]

print("Predictions saved to 'prediction.txt'.")

# Step 13: Calculate accuracy percentage
correct_count = (y_pred == y_test).sum()
incorrect_count = len(y_test) - correct_count
accuracy_percentage = (correct_count / len(y_test)) * 100

# Step 14: Print dataset and split information
print("\nDataset Summary:")
print(f"Total number of reviews in CSV: {len(df)}")
print(f"Number of reviews used for training: {X_train.shape[0]}")
print(f"Number of reviews used for testing: {X_test.shape[0]}")

# Step 15: Print prediction results
print("\nPrediction Results:")
print(f"Number of Correct Predictions: {correct_count}")
print(f"Number of Incorrect Predictions: {incorrect_count}")
print(f"Accuracy of the Model: {accuracy_percentage:.2f}%")

# Step 16: Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['1', '2', '3', '4', '5']))