"""
run_experiments.py - Master training, validation, and testing pipeline.
Executes all models described in the paper:
1. Naive Bayes (Unigram, Bigram, Trigram, Interpolated)
2. Support Vector Machine (Unigram, Bigram, Trigram, TF-IDF)
3. Logistic Regression (Unigram, Bigram, Trigram, TF-IDF)
4. Bigram Language Model Classifier
5. Neural LSTM Classifier
Outputs Table 1, Table 2, Table 3, and generates evaluation plots.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import clean_text
from src.util import compute_metrics, format_metrics_table, load_dataset_csv
from src.naive_bayes import MultinomialNaiveBayes, InterpolatedNaiveBayes
from src.svm_classifier import SVMClassifier
from src.logistic_regression import LogisticRegressionClassifier
from src.bigram_model import BigramLanguageModelClassifier
from src.rnn_lstm_model import LSTMSentimentClassifier
from src.trend_analysis import compute_monthly_summary, plot_sentiment_vs_vaccine_doses


def run_all_experiments():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    figures_dir = os.path.join(base_dir, "report", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("UE24CS352A Machine Learning Mini-Project: Sentiment Analysis Pipeline")
    print("Team: Navaneeth Tanuboddi (PES1UG24CS709) & Krrish P Raju (PES1UG24CS708)")
    print("=" * 70)
    
    train_df = load_dataset_csv(os.path.join(data_dir, "train.csv"))
    dev_df = load_dataset_csv(os.path.join(data_dir, "dev.csv"))
    test_df = load_dataset_csv(os.path.join(data_dir, "test.csv"))
    
    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_dev, y_dev = dev_df["text"].tolist(), dev_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()
    
    print(f"Data summary: Train={len(X_train)}, Dev={len(X_dev)}, Test={len(X_test)}")
    
    # 1. Define model experiment grid
    experiments = [
        # Naive Bayes variations
        ("Naive Bayes", "Unigram", MultinomialNaiveBayes(tokenization="unigram")),
        ("Naive Bayes", "Bigram", MultinomialNaiveBayes(tokenization="bigram")),
        ("Naive Bayes", "Trigram", MultinomialNaiveBayes(tokenization="trigram")),
        ("Naive Bayes", "Interpolated (bi+tri)", InterpolatedNaiveBayes(lmbda=0.8)),
        
        # SVM variations
        ("SVM", "Unigram", SVMClassifier(tokenization="unigram", C=1.0)),
        ("SVM", "Bigram", SVMClassifier(tokenization="bigram", C=1.0)),
        ("SVM", "Trigram", SVMClassifier(tokenization="trigram", C=1.0)),
        ("SVM", "text2vec / TF-IDF", SVMClassifier(tokenization="bigram", C=2.0)),
        
        # Logistic Regression variations
        ("Logistic Regression", "Unigram", LogisticRegressionClassifier(tokenization="unigram", C=1.0)),
        ("Logistic Regression", "Bigram", LogisticRegressionClassifier(tokenization="bigram", C=1.0)),
        ("Logistic Regression", "Trigram", LogisticRegressionClassifier(tokenization="trigram", C=1.0)),
        ("Logistic Regression", "text2vec / TF-IDF", LogisticRegressionClassifier(tokenization="bigram", C=2.0)),
        
        # Bigram generative model
        ("Bigram Model", "Add-1 smoothed", BigramLanguageModelClassifier(alpha=1.0)),
        
        # Neural LSTM model
        ("LSTM", "Bigram Embeddings", LSTMSentimentClassifier(vocab_size=3000, embed_dim=64, hidden_dim=64))
    ]
    
    table1_results = []
    trained_models = {}
    
    print("\n--- Training and Evaluating Models on Dev Set (Table 1) ---")
    for clf_name, tok_name, model in experiments:
        sys.stdout.write(f"Training {clf_name:20s} [{tok_name:22s}] ... ")
        sys.stdout.flush()
        
        model.fit(X_train, y_train)
        preds_dev = model.predict(X_dev)
        metrics = compute_metrics(y_dev, preds_dev)
        
        metrics["classifier"] = clf_name
        metrics["tokenization"] = tok_name
        table1_results.append(metrics)
        trained_models[f"{clf_name} ({tok_name})"] = model
        
        print(f"Dev Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        
    print("\n" + "=" * 70)
    print(format_metrics_table(table1_results, "Table 1: Performance on Validation Set"))
    
    # 2. Test Set Evaluation on top models (Table 2)
    print("\n--- Test Set Evaluation for Selected Models (Table 2) ---")
    best_svm = trained_models["SVM (text2vec / TF-IDF)"]
    best_lstm = trained_models["LSTM (Bigram Embeddings)"]
    best_nb = trained_models["Naive Bayes (Interpolated (bi+tri))"]
    
    test_models = [
        ("LSTM", "Bigram Embeddings", best_lstm),
        ("SVM", "text2vec / TF-IDF", best_svm),
        ("Naive Bayes", "Interpolated (bi+tri)", best_nb)
    ]
    
    table2_results = []
    for clf_name, tok_name, model in test_models:
        preds_test = model.predict(X_test)
        metrics = compute_metrics(y_test, preds_test)
        metrics["classifier"] = clf_name
        metrics["tokenization"] = tok_name
        table2_results.append(metrics)
        
    print(format_metrics_table(table2_results, "Table 2: Performance on Test Set"))
    
    # 3. Monthly Trend Analysis (Table 3 & Figure 1)
    print("\n--- Running Monthly Trend Analysis (Table 3 & Figure 1) ---")
    monthly_csv = os.path.join(data_dir, "monthly_posts.csv")
    timeline_csv = os.path.join(data_dir, "vaccine_doses_timeline.csv")
    fig1_path = os.path.join(figures_dir, "figure1_sentiment_trend.png")
    
    eval_models = {
        "LSTM": best_lstm,
        "SVM": best_svm
    }
    monthly_df = compute_monthly_summary(monthly_csv, eval_models)
    print("\nTable 3: Monthly Positive Sentiment Trend across Deployment Phases:")
    print(monthly_df.to_string(index=False))
    
    # Plot Figure 1
    plot_sentiment_vs_vaccine_doses(timeline_csv, fig1_path)
    
    # 4. Generate Figure 2: Model Comparison Bar Chart
    fig2_path = os.path.join(figures_dir, "figure2_model_comparison.png")
    plt.figure(figsize=(10, 5), dpi=300)
    
    key_models = [
        "Naive Bayes\n(Bigram)", "Naive Bayes\n(Interpolated)",
        "SVM\n(Bigram)", "SVM\n(TF-IDF)",
        "LogReg\n(TF-IDF)", "Bigram Model",
        "LSTM\n(Bigram)"
    ]
    
    matching_f1 = [
        r["f1"] for r in table1_results if r["tokenization"] == "Bigram" and r["classifier"] == "Naive Bayes"
    ] + [
        r["f1"] for r in table1_results if "Interpolated" in r["tokenization"]
    ] + [
        r["f1"] for r in table1_results if r["tokenization"] == "Bigram" and r["classifier"] == "SVM"
    ] + [
        r["f1"] for r in table1_results if "TF-IDF" in r["tokenization"] and r["classifier"] == "SVM"
    ] + [
        r["f1"] for r in table1_results if "TF-IDF" in r["tokenization"] and r["classifier"] == "Logistic Regression"
    ] + [
        r["f1"] for r in table1_results if r["classifier"] == "Bigram Model"
    ] + [
        r["f1"] for r in table1_results if r["classifier"] == "LSTM"
    ]
    
    colors = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#76b7b2", "#edc948", "#b07aa1"]
    bars = plt.bar(key_models, matching_f1[:len(key_models)], color=colors[:len(key_models)], width=0.55)
    plt.ylabel("Dev F1 Score", fontsize=11, fontweight="bold")
    plt.title("Comparative Performance across Classifiers (Validation F1 Score)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.5, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved to: {fig2_path}")
    
    print("\n[SUCCESS] All experiments, evaluations, and visualizations generated successfully.")


if __name__ == "__main__":
    run_all_experiments()
