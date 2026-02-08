# customer-support-ticket-nlp
Customer Support Ticket Cleaning and Classification using NLP
# 📌 Customer Support Ticket Cleaning & Classification System

An end-to-end NLP project that cleans, annotates, and classifies customer support tickets using **TF-IDF** and **Machine Learning**.

---

## 📖 Project Overview

Customer support tickets are noisy and unstructured, making automation difficult.  
This project builds a complete NLP pipeline to preprocess ticket text, extract features, and classify tickets into meaningful categories.

---

## 🎯 Objectives

- Clean and normalize raw ticket text
- Perform tokenization, lemmatization, and stopword removal
- Apply Named Entity Recognition (NER)
- Convert text into numerical features using TF-IDF
- Train and evaluate a machine learning classifier
- Visualize results using Confusion Matrix and ROC–AUC
- Analyze feature importance

---

## 🛠️ Tech Stack

- Python
- Pandas
- spaCy
- NLTK
- TextBlob
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📂 Project Structure

customer-support-ticket-nlp/
│
├── data/
│ ├── support_tickets.csv
│ └── processed_support_tickets.csv
│
├── models/
│ ├── ticket_classifier.pkl
│ └── tfidf_vectorizer.pkl
│
├── notebook/
│ └── customer_support_nlp.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md

**Machine Learning Model**

Feature Extraction: TF-IDF (Unigrams + Bigrams)

Classifier: Multinomial Naive Bayes

Evaluation: Accuracy, Precision, Recall, F1-score

Visualizations: Confusion Matrix, ROC–AUC

**Results**

Effective classification of support tickets

High interpretability using TF-IDF feature importance

Scalable pipeline suitable for real-world deployment


**Future Enhancements**

Use larger real-world datasets

Compare multiple classifiers (Logistic Regression, SVM)

Integrate Transformer-based models

Deploy as a REST API


**Author

Greeshma Yashmi**
NLP & Machine Learning Project
