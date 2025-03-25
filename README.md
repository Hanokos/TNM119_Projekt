
# Trip advisor review with explainability​
##### by: Hans Fredö, Daniel Fridström
##### date: 2025-03-25
#

## Installation & Setup

This project automatically installs missing dependencies from `requirements.txt`. If you haven't installed them manually, just run the script, and it will download the necessary libraries. After installation, **rerun the script** to ensure everything functions correctly.

Alternatively, you can manually install the required packages by running:

```sh
pip install -r requirements.txt
```

## Main Script: `main.py`
- The main script of the project is `main.py`. This file is used to:
- Calculate the accuracy of the models.
- Generate the pie chart in PNG format for model evaluation.
### After running this code in `Visual Studio code`, the script will print the model's accuracy in the terminal and save the pie chart as a PNG image.


## Demo Script: `demo.py`
### If you would like to test the model on your custom reviews, use the `demo.py` script. It allows you to input a review through a .txt file (e.g., `example.txt`) and get a predicted rating.
- 1: Create a .txt file with your review (e.g., example.txt):
- 2: Run the demo script:
### Test this with our premade `example.txt`
### After running this code in `Visual Studio code`, This will output a predicted rating based on the review in `example.txt`.
  
  
## Required Files
### This project requires the following `.pkl` and `.csv` files:

Make predictions on the hotel reviews and output the corresponding ratings.
### `Model3.pkl`
#### What it contains:
- A trained **XGBoost classifier** (`best_model3`) with optimized hyperparameters.
- Learned patterns (weights) to predict ratings (1-5) based on numerical features.

#### Purpose:
- Used to make predictions on new reviews **without retraining**.

#### Example Usage:
```python
import pickle
# Load and use the model
with open('Model3.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(new_review_features)  # Output: Rating (e.g., 2)
```

---
### `vectorizer3.pkl`
#### What it contains:
- A **fitted TF-IDF vectorizer** (`vectorizer3`) with vocabulary and weights.
- Stores the mapping of words/phrases to numerical features (e.g., "dirty" → Feature #123).

#### Purpose:
- Converts raw text into the same numerical format used during training.
- Ensures **new reviews** are processed **identically** to training data.

#### Example Usage:
```python
import pickle
# Load and use the vectorizer
with open('vectorizer3.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

new_review_features = vectorizer.transform(["The room was dirty"])  # Sparse matrix
```

### Key Takeaway
✅ **Model3.pkl** predicts ratings.  
✅ **vectorizer3.pkl** prepares text for the model.  
✅ **Both** are required for end-to-end functionality.  

---

## `tripadvisor_hotel_reviews.csv`
### What’s in the CSV?
File: **tripadvisor_hotel_reviews.csv**

#### Columns:
- **Review**: The hotel review text (e.g., *"The staff was rude, room was dirty."*).
- **Rating**: Star rating (*1 to 5*) given by the reviewer.

### How Your Code Uses It
#### Load Data:
```python
import pandas as pd
df = pd.read_csv('tripadvisor_hotel_reviews.csv')  # Loads into a Pandas DataFrame
```
Example:
| Review                                      | Rating |
|---------------------------------------------|--------|
| "The staff was rude..."                     | 2      |
| "Beautiful location, clean room."          | 5      |

#### Preprocess Text:
- Cleans the `Review` column (lowercase, remove URLs, etc.).
- Converts raw text → machine-readable format.

#### Extract Labels:
```python
y = df['Rating'] - 1  # Converts ratings (1-5) to (0-4) for XGBoost
```
**Why `-1`?** XGBoost expects classes starting at `0`.

#### Train/Test Split:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
- Uses **80%** of data for training, **20%** for testing.

---

## Summary:
📌 **main.py** for model accuracy and evaluation.
📌 **demo.py** for testing custom reviews.
📌 **CSV = Training data** (text + star ratings)  
📌 **Cleaned and split** for AI  
📌 **Model + Vectorizer** turn text into predictions  
📌 **End-to-end setup** for rating prediction! 🚀  



