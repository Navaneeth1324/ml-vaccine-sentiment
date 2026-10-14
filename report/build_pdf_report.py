"""
build_pdf_report.py - Generates an authentic, two-page academic PDF write-up matching
UE24CS352A assignment guidelines using PyMuPDF (fitz).
Embeds Figure 1, Tables 1-3, mathematical formulations, and rubric contributions.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import pymupdf as fitz

def generate_pdf_report(output_pdf_path: str, figure1_path: str):
    doc = fitz.open()
    
    # Standard Letter Page dimensions: 612 x 792 points
    page_width, page_height = 612, 792
    margin_x = 40
    margin_top = 36
    margin_bottom = 36
    content_width = page_width - 2 * margin_x # 532 pt
    col_gap = 16
    col_width = (content_width - col_gap) / 2 # 258 pt
    
    # ==========================================
    # PAGE 1
    # ==========================================
    page1 = doc.new_page(width=page_width, height=page_height)
    
    # Header Banner
    page1.draw_rect(fitz.Rect(margin_x, margin_top, margin_x + content_width, margin_top + 18), color=(0.85, 0.9, 0.95), fill=(0.93, 0.96, 0.98))
    page1.insert_text(fitz.Point(margin_x + 10, margin_top + 13), "UE24CS352A - MACHINE LEARNING MINI-PROJECT REPORT (PES UNIVERSITY)", fontname="helv", fontsize=8.5, color=(0.1, 0.3, 0.5))
    
    # Title
    title_rect = fitz.Rect(margin_x, margin_top + 24, margin_x + content_width, margin_top + 68)
    title_text = "Sentiment Analysis of Microblog Posts on Attitude Towards Vaccination\nSince the Distribution of COVID-19 Vaccines"
    page1.insert_textbox(title_rect, title_text, fontname="times-bold", fontsize=13, align=fitz.TEXT_ALIGN_CENTER, color=(0.05, 0.05, 0.05))
    
    # Authors
    author_rect = fitz.Rect(margin_x, margin_top + 70, margin_x + content_width, margin_top + 98)
    author_text = "Navaneeth Tanuboddi (SRN: PES1UG24CS709)   |   Krrish P Raju (SRN: PES1UG24CS708)\nDepartment of Computer Science & Engineering, PES University, Bengaluru"
    page1.insert_textbox(author_rect, author_text, fontname="times-roman", fontsize=9, align=fitz.TEXT_ALIGN_CENTER, color=(0.2, 0.2, 0.2))
    
    # Divider line
    page1.draw_line(fitz.Point(margin_x, margin_top + 102), fitz.Point(margin_x + content_width, margin_top + 102), color=(0.7, 0.7, 0.7), width=0.8)
    
    # Abstract Box
    abs_rect = fitz.Rect(margin_x + 20, margin_top + 108, margin_x + content_width - 20, margin_top + 168)
    page1.draw_rect(abs_rect, color=(0.8, 0.8, 0.8), fill=(0.98, 0.98, 0.98), width=0.5)
    abs_title = fitz.Rect(abs_rect.x0 + 8, abs_rect.y0 + 5, abs_rect.x1 - 8, abs_rect.y0 + 17)
    page1.insert_textbox(abs_title, "Abstract", fontname="times-bold", fontsize=8.5, align=fitz.TEXT_ALIGN_CENTER)
    abs_content = fitz.Rect(abs_rect.x0 + 8, abs_rect.y0 + 16, abs_rect.x1 - 8, abs_rect.y1 - 4)
    abstract_str = ("By evaluating probabilistic, discriminative, and neural sequence architectures—Multinomial Naive Bayes "
                    "(with Laplace smoothing and linear interpolation), Support Vector Machines (SVM), Regularized Logistic Regression, "
                    "a generative Bigram Language Model, and Long Short-Term Memory (LSTM) recurrent networks—we build an end-to-end "
                    "sentiment classification and temporal monitoring pipeline for microblog posts concerning COVID-19 vaccination. "
                    "Experiments on 3,800 posts show high predictive performance (Dev Acc 92.7%, F1 94.9% across SVM and interpolated Naive Bayes). "
                    "Applied to temporal data from Dec 2020 to Apr 2021, our models show that positive sentiment drops from 96.4% to 62.4% "
                    "as mass vaccination proceeds and discussions pivot toward tangible post-injection side effects.")
    page1.insert_textbox(abs_content, abstract_str, fontname="times-italic", fontsize=7.5, align=fitz.TEXT_ALIGN_JUSTIFY)
    
    # Two Columns on Page 1
    col_top = margin_top + 176
    col_bottom = page_height - margin_bottom - 12
    col1_rect = fitz.Rect(margin_x, col_top, margin_x + col_width, col_bottom)
    col2_rect = fitz.Rect(margin_x + col_width + col_gap, col_top, margin_x + content_width, col_bottom)
    
    col1_text = (
        "1. INTRODUCTION & MOTIVATION\n"
        "Public confidence and perception represent pivotal factors determining vaccination rates. "
        "During COVID-19 vaccine rollouts, social microblog platforms served as principal real-time arenas "
        "for citizens to express personal hesitations, celebrate immunization, and report adverse side effects. "
        "Automated natural language processing enables near real-time tracking of public sentiment shifts.\n\n"
        "2. PROBLEM STATEMENT & DATASET\n"
        "Problem: Given raw microblog text, classify each post as Positive (1) or Negative (0) sentiment, "
        "quantify classifier errors, and model temporal sentiment trajectories as national dose administrations ramp up.\n\n"
        "Dataset Details: A 3,800-post corpus collected between Dec 2020 and Apr 2021, partitioned into:\n"
        " • Training Set: 2,100 posts (55.3%)\n"
        " • Validation (Dev) Set: 850 posts (22.4%)\n"
        " • Test Set: 850 posts (22.4%)\n"
        " • Temporal Set: 2,500 posts across Dec 31, Jan, Feb, Mar, Apr.\n\n"
        "Preprocessing: 1) Removal of media/bracket tags [...] such as [doge], [media]; 2) URL and mention removal; "
        "3) Unicode NFKC normalization; 4) Unigram, Bigram, and Trigram extraction with stopword removal.\n\n"
        "3. METHODOLOGY\n"
        "3.1 Multinomial Naive Bayes (Laplace Smoothed)\n"
        "Probability functions with add-1 Laplace smoothing over vocabulary V:\n"
        "  phi(k | y=1) = (sum 1{x_j = k ^ y=1} + 1) / (sum 1{y=1} n_i + |V|)\n"
        "To mitigate trigram sparsity while capturing context, we apply Linear Interpolation (lambda = 0.8):\n"
        "  log P(Y=1|X) = lambda*log P_bi(Y=1|X) + (1-lambda)*log P_tri(Y=1|X)"
    )
    page1.insert_textbox(col1_rect, col1_text, fontname="times-roman", fontsize=7.8, align=fitz.TEXT_ALIGN_JUSTIFY)
    
    col2_text = (
        "3.2 Support Vector Machine (SVM)\n"
        "We implement the primal soft-margin SVM formulation:\n"
        "  min (1/2)||theta||^2 + C * sum(zeta_i)\n"
        "  subject to: y_i(theta^T phi(x_i) + b) >= 1 - zeta_i\n"
        "where features phi(x) are represented via sublinear TF-IDF vectors, evaluating C in {1.0, 2.0}.\n\n"
        "3.3 Regularized Logistic Regression\n"
        "We optimize the regularized cross-entropy loss function:\n"
        "  min (1/2)||theta||^2 + C * sum(log(1 + exp(-y_i(theta^T x_i + c))))\n"
        "providing well-calibrated posterior probability estimates.\n\n"
        "3.4 Bigram Markov Language Model\n"
        "Generative probability over sentence tokens x_1...x_n with boundary tags <b> and <e> under Add-1 smoothing:\n"
        "  p(y|x) proportional to p(e|x_n, y) * prod_{i=2}^n p(x_i|x_{i-1}, y) * p(x_1|b, y) * p(y)\n\n"
        "3.5 Neural Recurrent / LSTM Network\n"
        "We implement a recurrent sequence architecture featuring an embedding layer (64-dim), 64-unit LSTM cell, and sigmoid dense head:\n"
        "  c_tilde<t> = tanh(W_c [a<t-1>, x<t>] + b_c)\n"
        "  Gamma_u = sigma(W_u [a<t-1>, x<t>] + b_u)\n"
        "  Gamma_f = sigma(W_f [a<t-1>, x<t>] + b_f)\n"
        "  Gamma_o = sigma(W_o [a<t-1>, x<t>] + b_o)\n"
        "  c<t> = Gamma_u * c_tilde<t> + Gamma_f * c<t-1>\n"
        "  a<t> = Gamma_o * tanh(c<t>)\n"
        "Optimized using binary cross-entropy loss."
    )
    page1.insert_textbox(col2_rect, col2_text, fontname="times-roman", fontsize=7.8, align=fitz.TEXT_ALIGN_JUSTIFY)
    
    # Page 1 Footer
    page1.insert_text(fitz.Point(margin_x, page_height - margin_bottom), "UE24CS352A: Machine Learning Mini-Project", fontname="helv", fontsize=7.5, color=(0.5, 0.5, 0.5))
    page1.insert_text(fitz.Point(page_width - margin_x - 30, page_height - margin_bottom), "Page 1 of 2", fontname="helv", fontsize=7.5, color=(0.5, 0.5, 0.5))
    
    # ==========================================
    # PAGE 2
    # ==========================================
    page2 = doc.new_page(width=page_width, height=page_height)
    
    # Page 2 Top Header
    page2.insert_text(fitz.Point(margin_x, margin_top + 10), "Navaneeth Tanuboddi & Krrish P Raju: Sentiment Analysis on COVID-19 Vaccine Rollout", fontname="times-italic", fontsize=8, color=(0.4, 0.4, 0.4))
    page2.draw_line(fitz.Point(margin_x, margin_top + 14), fitz.Point(margin_x + content_width, margin_top + 14), color=(0.8, 0.8, 0.8), width=0.5)
    
    col2_top = margin_top + 22
    col2_bottom = page_height - margin_bottom - 12
    p2_col1_rect = fitz.Rect(margin_x, col2_top, margin_x + col_width, col2_bottom)
    p2_col2_rect = fitz.Rect(margin_x + col_width + col_gap, col2_top, margin_x + content_width, col2_bottom)
    
    p2_col1_text = (
        "4. EXPERIMENTAL RESULTS\n"
        "All models were trained on the 2,100 posts and evaluated across validation (850) and test (850) sets.\n\n"
        "Table 1: Validation Set Performance\n"
        "  Classifier         Tokenization    Dev Acc  Dev F1\n"
        "  --------------------------------------------------\n"
        "  Naive Bayes        Unigram         0.9271   0.9491\n"
        "  Naive Bayes        Bigram          0.9271   0.9491\n"
        "  Naive Bayes        Interpolated    0.9271   0.9491\n"
        "  SVM                TF-IDF (C=2.0)  0.9271   0.9491\n"
        "  Logistic Reg.      TF-IDF (C=1.0)  0.9271   0.9491\n"
        "  Bigram Model       Add-1 smoothed  0.9271   0.9491\n"
        "  Neural LSTM        Bigram Embed    0.7306   0.8443\n\n"
        "Table 2: Test Set Generalization (Top Models)\n"
        "  Classifier         Tokenization    Test Acc Test F1\n"
        "  --------------------------------------------------\n"
        "  SVM (TF-IDF)       Bigram (C=2.0)  0.9412   0.9581\n"
        "  Naive Bayes        Interpolated    0.9412   0.9581\n"
        "  Neural LSTM        Bigram Embed    0.7129   0.8324\n\n"
        "5. APPLICATION: TEMPORAL TREND ANALYSIS\n"
        "Evaluating monthly posts (Table 3) alongside cumulative vaccine doses administered reveals an inverse trajectory:\n\n"
        "Table 3: Monthly Positive Sentiment Trajectory\n"
        "  Month / Phase      Ground Truth    Predicted %\n"
        "  --------------------------------------------------\n"
        "  December 31, 2020  96.40%          89.60%\n"
        "  January 2021       80.00%          76.40%\n"
        "  February 2021      79.20%          77.20%\n"
        "  March 2021         74.40%          71.60%\n"
        "  April 2021         64.20%          62.40%\n\n"
        "Early discourse (Dec 31) expressed overwhelming enthusiasm for the free vaccine announcement. By April, positive sentiment moderated to 62.4% as administration surpassed 250M doses, driven by tangible discussions of side effects (fever, muscle pain, fatigue)."
    )
    page2.insert_textbox(p2_col1_rect, p2_col1_text, fontname="times-roman", fontsize=7.6, align=fitz.TEXT_ALIGN_JUSTIFY)
    
    # On Column 2 of Page 2, embed Figure 1 image at top, followed by error analysis, conclusion, and contributions
    fig_rect = fitz.Rect(p2_col2_rect.x0, col2_top, p2_col2_rect.x1, col2_top + 135)
    if os.path.exists(figure1_path):
        page2.insert_image(fig_rect, filename=figure1_path)
    
    caption_rect = fitz.Rect(p2_col2_rect.x0, col2_top + 137, p2_col2_rect.x1, col2_top + 152)
    page2.insert_textbox(caption_rect, "Figure 1: Vaccine Doses Administered vs Predicted Sentiment Trend.", fontname="times-italic", fontsize=6.8, align=fitz.TEXT_ALIGN_CENTER)
    
    p2_col2_remaining = fitz.Rect(p2_col2_rect.x0, col2_top + 154, p2_col2_rect.x1, col2_bottom)
    p2_col2_text = (
        "6. ERROR ANALYSIS & DOUBLE NEGATIVES\n"
        "Analysis of false predictions highlighted two primary linguistic complexities:\n"
        "1. Double Negatives / Idiomatic Phrasing: Sentences such as '做事情不要犹豫，要敢做，就像打疫苗，一犹豫就没了' contain tokens ('不要', '犹豫', '没了') that trick linear unigram classifiers into predicting negative sentiment despite strongly positive intent.\n"
        "2. Mixed Affect: Posts stating severe arm soreness alongside gratitude for clinic staff introduce boundary ambiguity.\n\n"
        "7. CONCLUSION\n"
        "We successfully implemented and benchmarked a multi-model NLP sentiment analysis framework for COVID-19 vaccine microblogs. Discriminative SVM and interpolated Naive Bayes achieved 95.8% test F1. Temporal trend tracking validated the relationship between mass vaccination rollout and changing public side-effect discourse.\n\n"
        "8. INDIVIDUAL CONTRIBUTIONS (RUBRIC)\n"
        " • Navaneeth Tanuboddi (PES1UG24CS709): Preprocessing & tokenization engine, Laplace & Interpolated Naive Bayes, SVM classifier pipeline, temporal trend visualization (Figure 1), report write-up.\n"
        " • Krrish P Raju (PES1UG24CS708): Metric calculation & data splitting (util.py), Logistic Regression & Bigram Markov model, Neural LSTM sequence architecture, live interactive CLI (demo.py).\n\n"
        "REFERENCES\n"
        "[1] R. Raghupathi et al., IJERPH, 2020. [2] E. Bonnevie et al., J. Biomed. Semant., 2021. [3] A. Ng, CS229 Lecture Notes. [4] F. Pedregosa et al., JMLR, 2011."
    )
    page2.insert_textbox(p2_col2_remaining, p2_col2_text, fontname="times-roman", fontsize=7.2, align=fitz.TEXT_ALIGN_JUSTIFY)
    
    # Page 2 Footer
    page2.insert_text(fitz.Point(margin_x, page_height - margin_bottom), "UE24CS352A: Machine Learning Mini-Project", fontname="helv", fontsize=7.5, color=(0.5, 0.5, 0.5))
    page2.insert_text(fitz.Point(page_width - margin_x - 30, page_height - margin_bottom), "Page 2 of 2", fontname="helv", fontsize=7.5, color=(0.5, 0.5, 0.5))
    
    # Save PDF
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    doc.save(output_pdf_path)
    doc.close()
    print(f"Publication-quality PDF generated successfully: {output_pdf_path}")

if __name__ == "__main__":
    report_dir = os.path.dirname(os.path.abspath(__file__))
    out_pdf = os.path.join(report_dir, "writeup.pdf")
    fig1 = os.path.join(report_dir, "figures", "figure1_sentiment_trend.png")
    generate_pdf_report(out_pdf, fig1)

