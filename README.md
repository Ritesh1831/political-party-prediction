# 🧠 Political Party Prediction Using Neural Networks

This project uses deep learning (TensorFlow/Keras) to predict whether a U.S. House representative is a **Democrat** or **Republican** based on their voting patterns from the 1984 Congressional Voting Records dataset.

---

## 📊 Dataset

- **Name**: 1984 Congressional Voting Records
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/congressional+voting+records)
- **Features**: 16 key legislative votes per member
- **Target**: Political party (`Democrat` or `Republican`)

---

## ⚙️ Models Used

Two deep learning approaches were explored:

### 1️⃣ Basic KerasClassifier with Cross-Validation

- Built using a `Sequential` model with three dense layers and dropout regularization.
- Applied 10-fold cross-validation using `cross_val_score`.
- Inputs include all 17 features (post-preprocessing).

### 2️⃣ Pipeline Model with Preprocessing

- Removes one categorical feature column (e.g. position or index column).
- Includes a `StandardScaler` + `KerasClassifier` in a `Pipeline`.
- Trains using `train_test_split` and evaluates with cross-validation and predictions on the test set.

---

## 🧠 Model Architecture

```text
Input → Dense(64, relu) → Dense(64, relu) → Dropout(0.5) → 
Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)
```
---

## File Structure

📦 PoliticalPartyPrediction/
├── PoliticalPartyPrediction.ipynb       # Jupyter Notebook
├── house-votes-84.names.txt             # Dataset description
├── README.md                            # Project documentation

---

## 🧪 Requirements

tensorflow
scikeras
scikit-learn
numpy
pandas

---

#### - Ritesh Raj

