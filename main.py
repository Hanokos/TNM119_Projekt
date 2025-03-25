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

# Step 1: Install and check required libraries
def install_requirements():
    try:
        import pandas
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
        import nltk
    except ImportError as e:
        print(f"Error: {e}")
        print("Installing missing libraries...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', 'scikit-learn', 'nltk', 'pandas', 'shap', 'matplotlib'])
        print("Installation complete. PLEASE RESTART THE SCRIPT!")
        sys.exit()
install_requirements()

nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load data
df = pd.read_csv('tripadvisor_hotel_reviews.csv')
y = df['Rating'] - 1


## Data Preprocessing & EDA
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['Review'] = df['Review'].apply(preprocess_text)


## Feature Engineering (TF-IDF)
# Step 3: Feature extraction
vectorizer3 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer3.fit_transform(df['Review'])

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# IF we have the model already reuse it
if os.path.exists('Model3.pkl') and os.path.exists('vectorizer3.pkl'):
    with open('Model3.pkl', 'rb') as f:
        best_model3 = pickle.load(f)
    with open('vectorizer3.pkl', 'rb') as f:
        vectorizer3 = pickle.load(f)
else: ## otherwise, do MODEL TRAINING
    param_grid = {'n_estimators': [100], 'max_depth': [5], 'learning_rate': [0.1], 'subsample': [0.8], 'colsample_bytree': [0.8]}
    model = XGBClassifier(random_state=42, objective='multi:softprob', num_class=5)
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=1, cv=StratifiedKFold(n_splits=3), scoring='accuracy', n_jobs=1, random_state=42)
    random_search.fit(X_train, y_train)
    best_model3 = random_search.best_estimator_
    with open('Model3.pkl', 'wb') as f: ## Deployment Prep, saves teh trained model and the vectorizer for reuse
        pickle.dump(best_model3, f)
    with open('vectorizer3.pkl', 'wb') as f:
        pickle.dump(vectorizer3, f)

y_pred = best_model3.predict(X_test)
accuracy_percentage = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")
print(classification_report(y_test, y_pred, target_names=['1', '2', '3', '4', '5']))

## Bias Testing & Explainability
# Step 5: SHAP visualization fix
explainer = shap.TreeExplainer(best_model3)
shap_values = explainer.shap_values(X_test)

# Aggregate absolute SHAP values across all classes
shap_values_abs_sum = sum(abs(shap_values[i]) for i in range(len(shap_values)))
feature_names = vectorizer3.get_feature_names_out()

shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values_abs_sum.mean(axis=0)})
shap_df = shap_df.sort_values(by='SHAP Value', ascending=False).head(20)

plt.figure(figsize=(10, 6))
plt.barh(shap_df['Feature'], shap_df['SHAP Value'], color='lightgreen')
plt.xlabel('SHAP Value (Impact on Prediction)')
plt.ylabel('Feature')
plt.title('Local Explanation (Top 20 Features)')
plt.gca().invert_yaxis()
plt.savefig('local_explanation.png')
plt.show()

print("Visualization saved as 'local_explanation.png'.")
