"""
error_analysis.py - Qualitative and Linguistic Error Analysis on Model Predictions.
Investigates failure modes across microblog sentiment classifiers:
1. Negation Reversals (e.g., 'not bad', '没有任何不良反应' vs '不安全')
2. Mixed Affect & Contrastive Conjunctions (e.g., '虽然...但是...', '打完后有点发烧，不过还好')
3. Rhetorical Expressions and Hesitancy Nuances
4. High-Confidence Mistakes (overconfident false predictions)

Author: Navaneeth Tanuboddi (PES1UG24CS709)
Course: UE24CS352A - Machine Learning Mini-Project, PES University
"""

import os
import sys
import csv
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, tokenize
from src.naive_bayes import MultinomialNaiveBayes, InterpolatedNaiveBayes
from src.bigram_model import BigramLanguageModelClassifier

# Linguistic cue sets for failure mode classification
NEGATION_TOKENS = {
    "不", "没", "没有", "别", "未", "无", "非", "莫", "休",
    "not", "no", "never", "without", "hardly", "barely", "scarcely", "neither", "nor"
}

CONTRAST_TOKENS = {
    "但是", "但", "不过", "然而", "可是", "只是", "虽然", "尽管",
    "but", "however", "although", "though", "yet", "despite", "nevertheless"
}

RHETORICAL_TOKENS = {
    "难道", "真的吗", "凭什么", "谁敢", "怎么可能", "？", "?", "！", "!"
}


def load_dataset_records(csv_path: str) -> List[Dict[str, Any]]:
    """Loads CSV dataset safely into a list of dicts using standard csv module."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "label": int(row.get("label", 0)),
                "sentiment": row.get("sentiment", "unknown"),
                "month": row.get("month", "")
            })
    return records


def detect_linguistic_features(text: str) -> Dict[str, bool]:
    """Detects presence of negation, contrastive conjunctions, and rhetorical markers."""
    cleaned = clean_text(text, remove_bracket_tags=False)
    lower_text = cleaned.lower()
    
    has_negation = any(tok in lower_text for tok in NEGATION_TOKENS)
    has_contrast = any(tok in lower_text for tok in CONTRAST_TOKENS)
    has_rhetorical = any(tok in lower_text for tok in RHETORICAL_TOKENS)
    
    return {
        "has_negation": has_negation,
        "has_contrast": has_contrast,
        "has_rhetorical": has_rhetorical
    }


def categorize_error(text: str, true_label: int, pred_label: int) -> str:
    """Categorizes why a model likely misclassified the microblog post."""
    features = detect_linguistic_features(text)
    
    if features["has_negation"] and features["has_contrast"]:
        return "Mixed Affect + Negation"
    elif features["has_negation"]:
        return "Negation Reversal"
    elif features["has_contrast"]:
        return "Contrastive Shift (e.g. although/but)"
    elif features["has_rhetorical"]:
        return "Rhetorical / Sarcastic Cue"
    elif len(text.strip()) < 15:
        return "Short / Sparse Lexical Context"
    else:
        return "Subtle Lexical Ambiguity"


class ErrorAnalyzer:
    """
    Performs error analysis on trained sentiment classifiers.
    Identifies misclassifications and categorizes linguistic failure modes.
    """
    def __init__(self, models: Optional[Dict[str, Any]] = None):
        if models is not None:
            self.models = models
        else:
            # Default to built-in pure Python models
            self.models = {
                "Naive_Bayes_Unigram": MultinomialNaiveBayes(tokenization="unigram"),
                "Naive_Bayes_Interpolated": InterpolatedNaiveBayes(lmbda=0.8),
                "Bigram_Model": BigramLanguageModelClassifier(alpha=1.0)
            }
            # Attempt to add SVM if sklearn is available
            try:
                from src.svm_classifier import SVMClassifier
                self.models["SVM_TFIDF"] = SVMClassifier(tokenization="bigram", C=2.0)
            except Exception:
                pass

    def fit_models(self, train_records: List[Dict[str, Any]]):
        """Fits all models on the training corpus."""
        X_train = [r["text"] for r in train_records]
        y_train = [r["label"] for r in train_records]
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def analyze(self, dev_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs evaluation on dev set and extracts all misclassified instances with linguistic annotations.
        """
        X_dev = [r["text"] for r in dev_records]
        y_dev = [r["label"] for r in dev_records]

        # Generate predictions from all models
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X_dev)

        error_cases = []
        for i, record in enumerate(dev_records):
            true_y = y_dev[i]
            text = X_dev[i]

            model_errors = {
                m_name: (preds[m_name][i] != true_y)
                for m_name in self.models.keys()
            }

            # If any model made an error on this post
            if any(model_errors.values()):
                ling_feat = detect_linguistic_features(text)
                
                # Use the primary model (Interpolated NB or SVM) to determine main error category
                primary_name = "SVM_TFIDF" if "SVM_TFIDF" in self.models else "Naive_Bayes_Interpolated"
                primary_pred = preds[primary_name][i]
                err_category = categorize_error(text, true_y, primary_pred)

                error_cases.append({
                    "id": record.get("id", f"case_{i}"),
                    "text": text,
                    "true_label": true_y,
                    "true_sentiment": "Positive" if true_y == 1 else "Negative",
                    "primary_pred": primary_pred,
                    "primary_correct": not model_errors[primary_name],
                    "all_predictions": {m: preds[m][i] for m in self.models.keys()},
                    "error_category": err_category,
                    "has_negation": ling_feat["has_negation"],
                    "has_contrast": ling_feat["has_contrast"],
                    "has_rhetorical": ling_feat["has_rhetorical"],
                    "length": len(text)
                })

        return error_cases

    def summarize_errors(self, error_cases: List[Dict[str, Any]], total_samples: int) -> Dict[str, Any]:
        """Generates statistical distribution of linguistic error categories."""
        category_counts = Counter(e["error_category"] for e in error_cases)
        total_errors = len(error_cases)
        
        summary = {
            "total_samples": total_samples,
            "total_misclassified": total_errors,
            "overall_error_rate": (total_errors / total_samples) if total_samples > 0 else 0.0,
            "categories": {
                cat: {
                    "count": count,
                    "percentage_of_errors": (count / total_errors * 100) if total_errors > 0 else 0.0
                }
                for cat, count in category_counts.most_common()
            }
        }
        return summary

    def export_csv(self, error_cases: List[Dict[str, Any]], output_path: str):
        """Exports error cases to a CSV report for inclusion in academic writeup."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fieldnames = [
            "id", "text", "true_label", "true_sentiment", "primary_pred",
            "primary_correct", "error_category", "has_negation", "has_contrast",
            "has_rhetorical", "length"
        ]
        with open(output_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in error_cases:
                row = {k: c[k] for k in fieldnames}
                writer.writerow(row)


def run_pipeline(data_dir: str, output_csv: Optional[str] = None):
    """Executes the complete error analysis pipeline and prints an formatted report."""
    train_file = os.path.join(data_dir, "train.csv")
    dev_file = os.path.join(data_dir, "dev.csv")

    if output_csv is None:
        output_csv = os.path.join(data_dir, "..", "..", "report", "error_analysis_cases.csv")
        output_csv = os.path.normpath(output_csv)

    print("=" * 68)
    print("  QUALITATIVE & LINGUISTIC ERROR ANALYSIS (UE24CS352A)")
    print("  Author: Navaneeth Tanuboddi (PES1UG24CS709)")
    print("=" * 68)

    print(f"\n[+] Loading training set from {train_file} ...")
    train_records = load_dataset_records(train_file)
    print(f"    Loaded {len(train_records)} training posts.")

    print(f"[+] Loading validation / dev set from {dev_file} ...")
    dev_records = load_dataset_records(dev_file)
    print(f"    Loaded {len(dev_records)} validation posts.")

    analyzer = ErrorAnalyzer()
    print(f"[+] Fitting models ({', '.join(analyzer.models.keys())}) ...")
    analyzer.fit_models(train_records)
    print("    Models trained.")

    print("[+] Evaluating predictions and isolating misclassifications ...")
    error_cases = analyzer.analyze(dev_records)
    summary = analyzer.summarize_errors(error_cases, total_samples=len(dev_records))

    print(f"\n[+] Identified {len(error_cases)} posts where at least one model failed.")
    print(f"    Overall error pool rate: {summary['overall_error_rate'] * 100:.2f}% of validation set.\n")

    print("-" * 68)
    print(f"{'Linguistic Failure Category':<35} | {'Count':<7} | {'% of Errors':<12}")
    print("-" * 68)
    for cat, stats in summary["categories"].items():
        print(f"{cat:<35} | {stats['count']:<7} | {stats['percentage_of_errors']:>6.2f}%")
    print("-" * 68)

    # Export to CSV
    analyzer.export_csv(error_cases, output_csv)
    print(f"\n[+] Saved detailed error cases to: {output_csv}")

    # Print top 3 illustrative examples for presentation slides
    print("\n[+] Representative Failure Cases for Qualitative Review:")
    print("=" * 68)
    for i, case in enumerate(error_cases[:3]):
        print(f"Sample #{i+1} [ID: {case['id']}]")
        print(f"  Post:     \"{case['text']}\"")
        print(f"  Ground:   {case['true_sentiment']} ({case['true_label']})")
        print(f"  Category: {case['error_category']}")
        print(f"  Model Predictions: {case['all_predictions']}")
        print("-" * 68)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, "..", "data", "processed")
    run_pipeline(default_data_dir)
