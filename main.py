import subprocess
import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

# This code makes the two pkl files if they don't exist
# IF they exist they reuse them instead.
# When successfully run this code will show the accuracy of the models and also its prediction in prediciton3.txt that it make,
# as well as two png files to show its accuracy in a piechart and global features weights png

# Step 1: Check and install required libraries
def install_requirements():
    try:
        import pandas
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import nltk
    except ImportError as e:
        print(f"Error: {e}")
        print("Some required libraries are missing. Installing them now...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', 'scikit-learn', 'nltk', 'pandas', 'shap', 'matplotlib'])
        print("Libraries installed successfully. PLEASE RESTART THE SCRIPT!")
        sys.exit()

# Call the function to check and install libraries
install_requirements()

# Step 2: Download NLTK resources (only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Step 3: Load the data
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# Step 4: Map ratings from [1, 2, 3, 4, 5] to [0, 1, 2, 3, 4]
y = df['Rating'] - 1

# Step 5: Advanced text preprocessing
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

df['Review'] = df['Review'].apply(preprocess_text)

# Step 6: Feature extraction with n-grams
vectorizer3 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Reduced max_features and ngram_range
X = vectorizer3.fit_transform(df['Review'])

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 8: Check if Model3.pkl and vectorizer3.pkl exist
if os.path.exists('Model3.pkl') and os.path.exists('vectorizer3.pkl'):
    print("Loading pre-trained model and vectorizer...")
    with open('Model3.pkl', 'rb') as f:
        best_model3 = pickle.load(f)
    with open('vectorizer3.pkl', 'rb') as f:
        vectorizer3 = pickle.load(f)
else:
    # Step 9: Hyperparameter tuning for XGBoost (simplified grid)
    param_grid = {
        'n_estimators': [100],  # Reduced number of trees
        'max_depth': [5],       # Reduced depth
        'learning_rate': [0.1],  # Single learning rate
        'subsample': [0.8],     # Single subsample ratio
        'colsample_bytree': [0.8],  # Single feature subsampling
    }

    model = XGBClassifier(random_state=42, objective='multi:softprob', num_class=5)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=1,  # Reduced number of parameter settings sampled
        cv=StratifiedKFold(n_splits=3),
        scoring='accuracy',
        n_jobs=1,  # Use fewer CPU cores
        random_state=42
    )
    random_search.fit(X_train, y_train)  # Use smaller dataset for tuning

    # Best model after hyperparameter tuning
    best_model3 = random_search.best_estimator_

    # Save the model and vectorizer
    with open('Model3.pkl', 'wb') as f:
        pickle.dump(best_model3, f)
    with open('vectorizer3.pkl', 'wb') as f:
        pickle.dump(vectorizer3, f)
    print("Model and vectorizer saved to Model3.pkl and vectorizer3.pkl")

# Step 10: Prediction
y_pred = best_model3.predict(X_test)

# Step 11: Save predictions to a new text file (prediction3.txt)
# Overwrite if the file already exists
with open('prediction3.txt', 'w') as file:
    for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
        # Map predictions and actual ratings back to [1, 2, 3, 4, 5]
        pred_rating = pred + 1
        actual_rating = actual + 1
        # Check if the prediction is correct
        correct = "Correct" if pred == actual else "Incorrect"
        # Write to file
        file.write(f"{i + 1}: Prediction \"{pred_rating}\", Actual Rating \"{actual_rating}\", {correct}\n")
        # Debug print to verify correctness
        print(f"Debug: Prediction {pred_rating}, Actual {actual_rating}, Label: {correct}")

# Step 12: Calculate accuracy percentage
correct_count = (y_pred == y_test).sum()
incorrect_count = len(y_test) - correct_count
accuracy_percentage = (correct_count / len(y_test)) * 100

# Step 13: Print dataset and split information
print("\nDataset Summary:")
print(f"Total number of reviews in CSV: {len(df)}")
print(f"Number of reviews used for training: {X_train.shape[0]}")
print(f"Number of reviews used for testing: {X_test.shape[0]}")

# Step 14: Print prediction results
print("\nPrediction Results:")
print(f"Number of Correct Predictions: {correct_count}")
print(f"Number of Incorrect Predictions: {incorrect_count}")
print(f"Accuracy of the Model: {accuracy_percentage:.2f}%")

# Step 15: Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['1', '2', '3', '4', '5']))

# Step 16: Pie Chart for Accuracy
labels = ['Correct', 'Incorrect']
sizes = [correct_count, incorrect_count]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # Explode the "Correct" slice

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Prediction Accuracy')
# Overwrite if the file already exists
if os.path.exists('accuracy_pie_chart.png'):
    os.remove('accuracy_pie_chart.png')
plt.savefig('accuracy_pie_chart.png')  # Save the pie chart as an image
plt.show()

# Step 17: Visualize Global Feature Importance (Top 20 Features Only)
plt.figure(figsize=(10, 6))
sorted_idx_top = best_model3.feature_importances_.argsort()[-20:][::-1]
plt.barh(range(len(sorted_idx_top)), best_model3.feature_importances_[sorted_idx_top], align='center')
plt.yticks(range(len(sorted_idx_top)), [vectorizer3.get_feature_names_out()[i] for i in sorted_idx_top])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances (Global)')

# Overwrite if the file already exists
if os.path.exists('feature_importance.png'):
    os.remove('feature_importance.png')
plt.savefig('feature_importance.png')  # Save the feature importance plot as an image
plt.show()

# Step 19: Print completion message
print("Visualizations saved as 'accuracy_pie_chart.png' and 'feature_importance.png'.")
