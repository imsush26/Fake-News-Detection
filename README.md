# Fake-News-Detection

### Project Overview

The rapid spread of fake news across digital platforms has created a need for automated systems that can identify and filter misinformation efficiently. This project implements a Fake News Detection system using Natural Language Processing (NLP) and supervised machine learning algorithms to classify news articles as Real or Fake based on their textual content.

The project compares multiple machine learning models and highlights the trade-off between high evaluation metrics and real-world generalization, especially when dealing with short text such as news titles.

### Objectives

* Build an automated fake news detection model using NLP techniques

* Convert unstructured text into numerical features using TF-IDF

* Train and evaluate multiple supervised ML classifiers

* Compare model performance using standard evaluation metrics

* Analyze model behavior on short titles vs full articles


### Dataset

The project uses publicly available datasets: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

* True.csv – Contains real news articles

* Fake.csv – Contains fake news articles

### Each dataset includes:

* News title

* News text

### A binary label is assigned:

* 1 → Real news

* 0 → Fake news

### Tech Stack

* Programming Language: Python

* Libraries:

1) pandas

2) numpy

3) nltk

4) scikit-learn

5) matplotlib

### Project Workflow

Data Collection --> Text Preprocessing(cleaning, stopword removal) --> TF-IDF Vectorization --> Model Training(LR, RF, DT, MNB) --> Model Evaluation --> Prediction & Analysis


### Text Preprocessing

* Lowercasing

* Removal of punctuation and numbers

* Stopword removal

* Tokenization

These steps reduce noise and improve model accuracy.

### Feature Extraction

* TF-IDF (Term Frequency–Inverse Document Frequency) is used to:

1) Convert text into numerical vectors

2) Assign higher importance to informative words

3) Reduce the impact of common words

### Machine Learning Models Used

* Logistic Regression

* Random Forest Classifier

* Decision Tree Classifier

* Multinomial Naive Bayes

### Model Performance (Test Data)

* Logistic Regression	      Accuracy=99.18%	Precision=98.87%	 Recall=99.44%	F1-Score=99.16%
* Random Forest	            Accuracy99.85%	Precision=99.90%	 Recall=99.79%	F1-Score=99.85%
* Decision Tree	            Accuracy=99.71%	Precision=99.70%	 Recall99.70%	F1-Score=99.70%
* Multinomial Naive Bayes	Accuracy=94.21%	Precision=94.40%	 Recall=93.53%	F1-Score93.97%

### Key Observations

* Random Forest achieved the highest overall accuracy and F1-score on full news articles

* Decision Tree and Logistic Regression also performed exceptionally well

* Multinomial Naive Bayes, despite lower overall accuracy, showed better generalization on short text inputs (news titles)

* High evaluation scores do not always guarantee correct real-world predictions on sparse or unseen inputs
