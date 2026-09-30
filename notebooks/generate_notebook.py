"""
generate_notebook.py - Generates the interactive walkthrough notebook for the demonstration.
"""
import json
import os

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# UE24CS352A - Machine Learning Mini-Project\n",
    "## Sentiment Analysis of Microblog Posts on Attitude Towards Vaccination Since the Distribution of COVID-19 Vaccines\n",
    "\n",
    "**Team Members:**\n",
    "- **Navaneeth Tanuboddi** (SRN: `PES1UG24CS709`)\n",
    "- **Krrish P Raju** (SRN: `PES1UG24CS708`)\n",
    "\n",
    "---\n",
    "### Project Abstract & Objectives\n",
    "In this project, we reproduce and implement the machine learning pipeline for microblog sentiment analysis on COVID-19 vaccination:\n",
    "1. **Preprocessing & N-gram Tokenization** (Unigram, Bigram, Trigram, TF-IDF)\n",
    "2. **Classical Classifiers**: Naive Bayes (Laplace smoothing & linear interpolation), Support Vector Machines (SVM), and Regularized Logistic Regression\n",
    "3. **Generative Modeling**: Bigram Language Model Classifier\n",
    "4. **Deep Sequence Modeling**: Recurrent Neural Network / LSTM with embedding layers\n",
    "5. **Temporal Trend Analysis**: Tracking public sentiment evolution against national vaccine administration trajectories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Add source path\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "from src.preprocessing import clean_text, tokenize\n",
    "from src.util import compute_metrics, format_metrics_table, load_dataset_csv\n",
    "from src.naive_bayes import MultinomialNaiveBayes, InterpolatedNaiveBayes\n",
    "from src.svm_classifier import SVMClassifier\n",
    "from src.logistic_regression import LogisticRegressionClassifier\n",
    "from src.bigram_model import BigramLanguageModelClassifier\n",
    "from src.rnn_lstm_model import LSTMSentimentClassifier\n",
    "\n",
    "print(\"All modules loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Load Dataset Splits\n",
    "We load the curated dataset splits (Train: 2100, Dev: 850, Test: 850) matching the reference study."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = '../data/processed'\n",
    "train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))\n",
    "dev_df = pd.read_csv(os.path.join(data_dir, 'dev.csv'))\n",
    "test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))\n",
    "\n",
    "print(f\"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}\")\n",
    "train_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Preprocessing & N-Gram Tokenization Demo\n",
    "Demonstrating clean_text, unigram, bigram, and trigram feature extraction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_post = \"今天去社区医院接种了新冠疫苗，整体感觉很棒，除了针口稍微有点胀痛，完全没有其他异常反应，点赞！\"\n",
    "cleaned = clean_text(sample_post)\n",
    "print(\"Cleaned Text:\", cleaned)\n",
    "print(\"Unigram tokens:\", tokenize(cleaned, method='unigram')[:8])\n",
    "print(\"Bigram tokens:\", tokenize(cleaned, method='bigram')[:8])\n",
    "print(\"Trigram tokens:\", tokenize(cleaned, method='trigram')[:8])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Model Training & Validation (Table 1 Replication)\n",
    "Training Naive Bayes, SVM, Logistic Regression, Bigram Model, and LSTM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, y_train = train_df['text'].tolist(), train_df['label'].tolist()\n",
    "X_dev, y_dev = dev_df['text'].tolist(), dev_df['label'].tolist()\n",
    "\n",
    "# Initialize models\n",
    "models = {\n",
    "    'Naive Bayes (Bigram)': MultinomialNaiveBayes(tokenization='bigram'),\n",
    "    'Naive Bayes (Interpolated)': InterpolatedNaiveBayes(lmbda=0.8),\n",
    "    'SVM (TF-IDF)': SVMClassifier(tokenization='bigram', C=2.0),\n",
    "    'Logistic Regression': LogisticRegressionClassifier(tokenization='bigram', C=1.0),\n",
    "    'Bigram Model': BigramLanguageModelClassifier(alpha=1.0),\n",
    "    'Neural LSTM': LSTMSentimentClassifier(vocab_size=3000, embed_dim=64, hidden_dim=64)\n",
    "}\n",
    "\n",
    "results = []\n",
    "for name, clf in models.items():\n",
    "    print(f\"Training {name}...\")\n",
    "    clf.fit(X_train, y_train)\n",
    "    preds = clf.predict(X_dev)\n",
    "    m = compute_metrics(y_dev, preds)\n",
    "    m['Model'] = name\n",
    "    results.append(m)\n",
    "\n",
    "res_df = pd.DataFrame(results)[['Model', 'accuracy', 'precision', 'recall', 'f1']]\n",
    "res_df.columns = ['Model', 'Dev Accuracy', 'Precision', 'Recall', 'F1 Score']\n",
    "res_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Temporal Trend Analysis & Figure 1 Plot\n",
    "Evaluating how public sentiment evolved across December 2020 through April 2021 as national vaccine doses were rolled out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.trend_analysis import plot_sentiment_vs_vaccine_doses\n",
    "\n",
    "timeline_csv = '../data/processed/vaccine_doses_timeline.csv'\n",
    "fig_out = '../report/figures/figure1_sentiment_trend.png'\n",
    "plot_sentiment_vs_vaccine_doses(timeline_csv, fig_out)\n",
    "\n",
    "img = plt.imread(fig_out)\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Interactive Live Inference Test\n",
    "Enter any microblog sentence below to run real-time sentiment predictions across models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_custom_text(text):\n",
    "    print(f\"Post: {text}\\n\" + \"-\" * 50)\n",
    "    for name, clf in models.items():\n",
    "        if hasattr(clf, 'predict_proba'):\n",
    "            p = clf.predict_proba(text)\n",
    "            p1 = p[1] if isinstance(p, dict) else (p[0][1] if hasattr(p, 'shape') else 0.5)\n",
    "            print(f\"{name:<26}: {'POSITIVE' if p1 >= 0.5 else 'NEGATIVE'} (confidence: {max(p1, 1-p1)*100:.1f}%)\")\n",
    "        else:\n",
    "            pred = clf.predict([text])[0]\n",
    "            print(f\"{name:<26}: {'POSITIVE' if pred==1 else 'NEGATIVE'}\")\n",
    "\n",
    "test_custom_text(\"打了第一针新冠疫苗，没有任何不良反应，相信国家！\")\n",
    "print()\n",
    "test_custom_text(\"打完疫苗后手臂酸痛了好几天，甚至抬不起来，难受。\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs("/Users/macm2/.gemini/antigravity/scratch/ml-vaccine-sentiment/notebooks", exist_ok=True)
with open("/Users/macm2/.gemini/antigravity/scratch/ml-vaccine-sentiment/notebooks/sentiment_analysis_walkthrough.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook generated successfully!")
