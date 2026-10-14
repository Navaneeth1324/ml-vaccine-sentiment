"""
demo.py - Interactive Live Demonstration Script for Review & Examination.
Designed for UE24CS352A Machine Learning Mini-Project.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import sys
import pandas as pd
from src.util import load_dataset_csv
from src.naive_bayes import MultinomialNaiveBayes, InterpolatedNaiveBayes
from src.svm_classifier import SVMClassifier
from src.logistic_regression import LogisticRegressionClassifier
from src.bigram_model import BigramLanguageModelClassifier
from src.rnn_lstm_model import LSTMSentimentClassifier
from src.fact_checker import FactCheckVerifier

fact_verifier = FactCheckVerifier()


def load_and_train_all(data_dir: str):
    print(">> Initializing and training models on training set (2100 posts)...")
    train_df = load_dataset_csv(os.path.join(data_dir, "train.csv"))
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()

    models = {
        "SVM (TF-IDF)": SVMClassifier(tokenization="bigram", C=2.0),
        "Naive Bayes (Interpolated)": InterpolatedNaiveBayes(lmbda=0.8),
        "Logistic Regression": LogisticRegressionClassifier(tokenization="bigram", C=1.0),
        "Bigram Model": BigramLanguageModelClassifier(alpha=1.0),
        "Neural LSTM": LSTMSentimentClassifier(vocab_size=3000, embed_dim=64, hidden_dim=64)
    }

    for name, model in models.items():
        sys.stdout.write(f"   [+] Training {name} ... ")
        sys.stdout.flush()
        model.fit(X_train, y_train)
        print("Ready.")
        
    return models


import argparse

# ANSI color codes for enhanced terminal readability
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


def predict_sample(models, text: str):
    print("\n" + "=" * 65)
    print(f"{COLOR_BOLD}INPUT POST:{COLOR_RESET} {text}")
    print("=" * 65)
    print(f"{'Model':<28} | {'Predicted Sentiment':<20} | {'Confidence / Prob':<15}")
    print("-" * 65)

    primary_pred = 0
    primary_conf = 0.5

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text)
            if isinstance(probs, dict):
                p_neg, p_pos = probs[0], probs[1]
            elif hasattr(probs, "shape") and len(probs.shape) == 2:
                p_neg, p_pos = probs[0][0], probs[0][1]
            else:
                p_neg, p_pos = 0.5, 0.5
            if p_pos >= 0.5:
                pred = f"{COLOR_GREEN}POSITIVE (1){COLOR_RESET}"
                pred_int = 1
                conf_val = p_pos
            else:
                pred = f"{COLOR_RED}NEGATIVE (0){COLOR_RESET}"
                pred_int = 0
                conf_val = p_neg
            conf = f"{max(p_neg, p_pos)*100:.2f}%"
        else:
            p = model.predict([text])[0]
            pred_int = int(p)
            conf_val = 0.85
            if p == 1:
                pred = f"{COLOR_GREEN}POSITIVE (1){COLOR_RESET}"
            else:
                pred = f"{COLOR_RED}NEGATIVE (0){COLOR_RESET}"
            conf = "N/A"

        if "Logistic" in name or "SVM" in name:
            primary_pred = pred_int
            primary_conf = conf_val

        print(f"{name:<28} | {pred:<29} | {conf:<15}")

    print("=" * 65)
    
    # Grounded Fact-Check Verification Engine output
    fc_result = fact_verifier.verify(text, sentiment_pred=primary_pred, sentiment_conf=primary_conf)
    
    risk_color = COLOR_GREEN if fc_result.risk_level == "Low" else (COLOR_YELLOW if fc_result.risk_level == "Moderate" else COLOR_RED)
    print(f"\n{COLOR_BOLD}[GROUNDED MISINFORMATION & FACT-CHECK VERIFICATION]{COLOR_RESET}")
    print(f"Sentiment: {COLOR_RED if primary_pred == 0 else COLOR_GREEN}{fc_result.sentiment_label} ({fc_result.sentiment_conf:.2f}){COLOR_RESET}")
    print(f"Flag: Misinformation Risk: {risk_color}{COLOR_BOLD}{fc_result.risk_level}{COLOR_RESET}")
    if fc_result.grounded_fact:
        print(f'Grounded Fact: "{fc_result.grounded_fact}"')
    else:
        print('Grounded Fact: None (No verified misinformation narrative detected.)')
    print("=" * 65 + "\n")



def interactive_cli():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    print("\n" + "=" * 70)
    print(f"  {COLOR_BOLD}COVID-19 VACCINE ATTITUDE SENTIMENT ANALYSIS - LIVE DEMO{COLOR_RESET}")
    print("  Course: UE24CS352A - Machine Learning Mini-Project")
    print("  Presenters: Navaneeth Tanuboddi (709) & Krrish P Raju (708)")
    print("=" * 70)
    
    models = load_and_train_all(data_dir)
    
    samples = [
        "打了第一针新冠疫苗，没有任何不良反应，相信祖国！(Positive)",
        "打完疫苗后手臂酸痛了好几天，甚至抬不起来，难受。(Negative)",
        "第二针搞定，医护人员辛苦了，中国加油！(Positive)",
        "听说有人接种后出现过敏反应，心里还是挺害怕的。(Negative)",
        "Just received my first COVID-19 vaccine dose! Smooth process and grateful. (Positive)",
        "Experienced severe fever and chills after the second shot. Really exhausted. (Negative)",
        "mRNA vaccines will alter human DNA and change your genetic code forever! (Misinformation: DNA)",
        "COVID-19 vaccines contain 5G tracking microchips to control the population! (Misinformation: Microchip)",
        "听说打新冠疫苗会改变人体DNA和基因结构，绝对不能打！(Misinformation: Chinese DNA)"
    ]

    while True:
        print("\nDemo Menu Options:")
        print("1. Test predefined sample posts (Positive, Negative, & Misinformation)")
        print("2. Enter custom microblog post text")
        print("3. View monthly sentiment trend report summary")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\nSelect a preset sample:")
            for i, s in enumerate(samples, 1):
                print(f"[{i}] {s}")
            idx_str = input(f"Enter sample number (1-{len(samples)}): ").strip()
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(samples):
                    raw_text = samples[idx].split("(")[0].strip()
                    predict_sample(models, raw_text)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid input.")
        elif choice == "2":
            user_text = input("\nEnter post text: ").strip()
            if user_text:
                predict_sample(models, user_text)
            else:
                print("Empty input.")
        elif choice == "3":
            timeline_csv = os.path.join(data_dir, "vaccine_doses_timeline.csv")
            if os.path.exists(timeline_csv):
                df = pd.read_csv(timeline_csv)
                print("\nNational Vaccine Rollout & Sentiment Trajectory:")
                print(df.to_string(index=False))
            else:
                print("Timeline data not found.")
        elif choice == "4":
            print("Exiting demo. Good luck with the review!")
            break
        else:
            print("Invalid choice, please select 1-4.")


def main():
    parser = argparse.ArgumentParser(description="Live Sentiment Analysis Demonstration CLI")
    parser.add_argument("--text", type=str, help="Evaluate sentiment for a single custom text string directly.")
    parser.add_argument("--preset", type=int, choices=range(1, 10), help="Run evaluation on preset sample index (1-9).")
    parser.add_argument("--report", action="store_true", help="Print national rollout and sentiment timeline report.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")

    samples = [
        "打了第一针新冠疫苗，没有任何不良反应，相信祖国！",
        "打完疫苗后手臂酸痛了好几天，甚至抬不起来，难受。",
        "第二针搞定，医护人员辛苦了，中国加油！",
        "听说有人接种后出现过敏反应，心里还是挺害怕的。",
        "Just received my first COVID-19 vaccine dose! Smooth process and grateful.",
        "Experienced severe fever and chills after the second shot. Really exhausted.",
        "mRNA vaccines will alter human DNA and change your genetic code forever!",
        "COVID-19 vaccines contain 5G tracking microchips to control the population!",
        "听说打新冠疫苗会改变人体DNA和基因结构，绝对不能打！"
    ]

    if args.text:
        models = load_and_train_all(data_dir)
        predict_sample(models, args.text)
    elif args.preset:
        models = load_and_train_all(data_dir)
        predict_sample(models, samples[args.preset - 1])
    elif args.report:
        timeline_csv = os.path.join(data_dir, "vaccine_doses_timeline.csv")
        if os.path.exists(timeline_csv):
            df = pd.read_csv(timeline_csv)
            print("\nNational Vaccine Rollout & Sentiment Trajectory:")
            print(df.to_string(index=False))
        else:
            print("Timeline data not found.")
    else:
        interactive_cli()



if __name__ == "__main__":
    main()

