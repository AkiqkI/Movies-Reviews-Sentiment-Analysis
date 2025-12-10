# 🎬 Movie Review Sentiment Analysis

## Project Overview

This repository hosts a machine learning project aimed at classifying movie reviews into positive or negative sentiments. The primary objective was to develop a model that achieves an F1 score of at least 0.85 in detecting negative reviews, utilizing a dataset of IMDB movie reviews with polarity labels.

## Dataset

The project uses a dataset of IMDB movie reviews, which includes text reviews and a `pos` column indicating polarity (1 for positive, 0 for negative). The dataset is pre-split into training and testing sets.

## Methodology

The project involved the following steps:

1.  **Data Loading and Initial Exploration:** Loaded the dataset and performed an initial exploratory data analysis (EDA) to understand distributions of reviews, ratings, and polarities over time.
2.  **Text Normalization:** Reviews were converted to lowercase and cleaned of digits and punctuation.
3.  **Model Experimentation:** Six different models were trained and evaluated:
    *   `model_0`: Dummy Classifier (constant baseline)
    *   `model_1`: NLTK, TF-IDF, and Logistic Regression
    *   `model_3`: spaCy, TF-IDF, and Logistic Regression
    *   `model_4`: spaCy, TF-IDF, and LGBMClassifier
    *   `model_9`: BERT embeddings and Logistic Regression (on 200 samples)
    *   `model_10`: BERT embeddings and LGBMClassifier (on 200 samples)
4.  **Evaluation Procedure:** A comprehensive `evaluate_model` function was created to assess models using Accuracy, F1 Score, Average Precision Score (APS), and ROC AUC, along with visualizing F1, ROC, and Precision-Recall curves.
5.  **Custom Review Validation:** The best-performing models were further tested on a set of manually crafted reviews to evaluate their generalization capabilities on unseen, real-world text.

## Results and Conclusions

The project successfully met its objective, with `Model 1` being the top performer. Below is a summary of the key models' performance on the test set:

| Model | Vectorizer | Algorithm | Test F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | **TF-IDF (NLTK)** | **LogReg** | **0.88** | 🏆 **Winner** |
| **Model 3** | TF-IDF (spaCy) | LogReg | 0.87 | ✅ Pass |
| **Model 4** | TF-IDF (spaCy) | LGBM | 0.85 | ✅ Pass (Borderline) |
| **Model 9** | BERT | LogReg | 0.78 | ❌ Fail (Insufficient Data) |
| **Model 10** | BERT | LGBM | 0.70 | ❌ Fail (Insufficient Data) |

### Key Findings:

*   **Simplicity Wins:** Logistic Regression models with TF-IDF features (both NLTK and spaCy preprocessors) significantly outperformed more complex tree-based and BERT models for this task, especially when BERT was constrained by limited data.
*   **NLTK vs. spaCy:** NLTK tokenizer combined with TF-IDF and Logistic Regression (`Model 1`) provided slightly better results than its spaCy counterpart (`Model 3`).
*   **BERT Limitations with Small Data:** While BERT embeddings showed strong discriminative power (high ROC AUC), the models trained on a limited subset (200 samples) suffered from severe overfitting and poor probability calibration, leading to lower F1 scores.
*   **Robust Generalization:** `Model 1` and `Model 3` demonstrated perfect classification on a set of custom, manually written reviews, confirming their strong generalization capabilities.

## Recommendation

**Model 1 (NLTK + TF-IDF + Logistic Regression)** is recommended for deployment due to its superior F1 score (0.88), excellent generalization ability, and computational efficiency.
