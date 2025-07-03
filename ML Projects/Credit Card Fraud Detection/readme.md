# 💳 Credit Card Fraud Detection using Logistic Regression

This project uses machine learning to detect fraudulent credit card transactions and provides a real-time prediction interface using Streamlit.

---

## 📂 Dataset

The dataset contains **284,807 transactions** with **31 columns**.  
The `Class` column is the target variable:

- `0`: Legitimate Transaction  
- `1`: Fraudulent Transaction

> ⚠️ Note: The dataset is highly **imbalanced**, with a very small number of fraud cases.

---

## 🧹 Preprocessing

- Separated legitimate and fraudulent transactions
- **Undersampled** legitimate transactions to balance the dataset
- Split data into **training** and **testing** sets using `train_test_split()`

---

## 🤖 Model: Logistic Regression

We used **Logistic Regression** from Scikit-learn:

- Trained on balanced data
- Used to classify transactions as legitimate or fraudulent
- Scaled data (if necessary) for better convergence

---

## 📈 Evaluation

Model performance was evaluated using **accuracy**:
- High accuracy on both training and testing sets
- Useful for detecting fraud in real-time environments

> You can improve the model further using metrics like **precision**, **recall**, **F1-score**, or **ROC-AUC**, especially on imbalanced datasets.

---

## 🖥️ Streamlit Web App

A **Streamlit interface** is provided for user interaction:

### 🔧 Features:
- Upload your own `.csv` dataset for training
- Enter feature values to check if a transaction is fraudulent
- Real-time prediction based on logistic regression

### ▶️ How to Run the App:

```bash
pip install -r requirements.txt
streamlit run app.py
