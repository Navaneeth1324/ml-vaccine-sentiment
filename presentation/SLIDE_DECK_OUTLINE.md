# UE24CS352A Machine Learning Mini-Project - Final Review Presentation
## Sentiment Analysis of Social Media Posts on Attitude Towards COVID-19 Vaccination

**Team Members:**
- **Navaneeth Tanuboddi** (SRN: `PES1UG24CS709`)
- **Krrish P Raju** (SRN: `PES1UG24CS708`)

---

## Slide 1: Title & Team Introduction
* **Title:** Sentiment Analysis of Microblog Posts on Attitude Towards Vaccination Since the Distribution of COVID-19 Vaccines
* **Course:** UE24CS352A - Machine Learning Mini-Project
* **Team:**
  - Navaneeth Tanuboddi (`PES1UG24CS709`)
  - Krrish P Raju (`PES1UG24CS708`)
* **Speaker:** Navaneeth Tanuboddi
* **Talking Points:**
  > "Good morning / afternoon professors and panel members. Today, Krrish and I will present our machine learning project analyzing public attitude and sentiment trajectories towards COVID-19 vaccination from microblog social media posts. We implemented classical probabilistic, discriminative, and neural sequence architectures to evaluate sentiment and correlated it with national vaccination rollout stages."

---

## Slide 2: Problem Statement & Motivation
* **Key Challenge:** Assessing public confidence, hesitancy, and concerns during mass vaccination campaigns using natural language processing.
* **Why Social Media / Microblogs?** High volume, real-time public reactions, direct reports of side effects, and policy reception.
* **Core Objectives:**
  1. Build an end-to-end NLP classification pipeline comparing generative (Naive Bayes, Bigram LM), discriminative (SVM, Logistic Regression), and deep neural (LSTM) models.
  2. Implement N-gram feature variations (Unigram, Bigram, Trigram, TF-IDF).
  3. Perform temporal sentiment tracking across successive months (Dec 2020 – Apr 2021) and correlate with vaccine dose administration.
* **Speaker:** Krrish P Raju

---

## Slide 3: Dataset & Preprocessing Pipeline
* **Data Characteristics:**
  - Microblog text samples collected across key deployment months (Dec 31, Jan, Feb, Mar, Apr).
  - Train Set: 2,100 posts | Dev Set: 850 posts | Test Set: 850 posts.
* **Preprocessing Operations:**
  - Discarding bracketed media tags `[...]` (emojis, image tags).
  - URL and mention stripping (`@user`, `#hashtag#`).
  - Unicode NFKC normalization.
  - Multi-granularity tokenization:
    - **Unigram:** Character/word level tokens.
    - **Bigram & Trigram:** Consecutive token tuples capturing local context.
    - **Stopword filtering:** Removing frequent non-informative grammatical particles.
* **Speaker:** Navaneeth Tanuboddi

---

## Slide 4: Mathematical Methodology (Part 1 - Probabilistic & Linear Models)
* **1. Multinomial Naive Bayes with Laplace Smoothing:**
  $$\Phi_{k|y=1} = \frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\{x_j^{(i)} = k \wedge y^{(i)} = 1\} + 1}{\sum_{i=1}^m 1\{y^{(i)}=1\}n_i + |V|}$$
* **Linear Interpolation:** Combining bigram and trigram predictions ($\lambda = 0.8$):
  $$\log P(Y=1|X) = \lambda \log P_{\text{bigram}}(Y=1|X) + (1-\lambda) \log P_{\text{trigram}}(Y=1|X)$$
* **2. Support Vector Machine (SVM):**
  - Primal soft-margin formulation: $\min_{\theta, b, \zeta} \frac{1}{2}\|\theta\|^2 + C \sum_{i=1}^n \zeta_i$ subject to $y_i(\theta^T \phi(x_i) + b) \ge 1 - \zeta_i$.
* **3. Regularized Logistic Regression:**
  - Minimizing regularized cross-entropy loss: $\min_{\theta, c} \frac{1}{2}\|\theta\|^2 + C \sum_{i=1}^n \log(1 + e^{-y_i(\theta^Tx_i + c)})$.
* **Speaker:** Krrish P Raju

---

## Slide 5: Mathematical Methodology (Part 2 - Bigram LM & LSTM Neural Network)
* **4. Generative Bigram Markov Model:**
  $$P(y|x) \propto P(e|x_n, y) \cdot \prod_{i=2}^n P(x_i|x_{i-1}, y) \cdot P(x_1|b, y) \cdot P(y)$$
* **5. Recurrent Neural Network / LSTM Architecture:**
  - Embedding Layer (vocab projection) $\to$ LSTM cell with 64 units $\to$ Dense Sigmoid head.
  - Cell state & gating equations:
    $$\tilde{c}^{\langle t \rangle} = \tanh(W_c [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$$
    $$\Gamma_u = \sigma(W_u [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u), \quad \Gamma_f = \sigma(W_f [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f), \quad \Gamma_o = \sigma(W_o [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o)$$
    $$c^{\langle t \rangle} = \Gamma_u \tilde{c}^{\langle t \rangle} + \Gamma_f c^{\langle t-1 \rangle}, \quad a^{\langle t \rangle} = \Gamma_o \tanh(c^{\langle t \rangle})$$
  - Loss: Binary cross-entropy optimization.
* **Speaker:** Navaneeth Tanuboddi

---

## Slide 6: Experimental Results & Model Comparison
* **Validation Performance (Table 1 Summary):**
  - Linear Interpolated Naive Bayes: Dev Acc **92.71%**, F1 **94.91%**
  - SVM with TF-IDF / Bigram: Dev Acc **92.71%**, F1 **94.91%**
  - Logistic Regression: Dev Acc **92.71%**, F1 **94.91%**
  - Neural LSTM: Dev Acc **73.06%**, F1 **84.43%**
* **Test Set Generalization (Table 2):**
  - Top discriminative models sustained test F1 of **95.81%**.
  - LSTM achieved test F1 of **83.24%** with high recall.
* **Speaker:** Krrish P Raju

---

## Slide 7: Temporal Trend Analysis (Figure 1 & Table 3)
* **Trajectory of Positive Posts vs Vaccine Rollout:**
  - **Dec 31, 2020:** ~95.6% Positive (high initial enthusiasm upon free vaccine announcement).
  - **Jan – Feb 2021:** ~77–80% Positive (steady uptake, community clinics open).
  - **March 2021:** ~74.4% Positive (expanded cohort).
  - **April 2021:** ~62.4% Positive (sharp increase in administered doses $\to$ wider public discussion of side effects like soreness, fatigue, fever).
* **Finding:** While positive sentiment remains predominant overall, confidence dips moderately as real-world side effects are experienced and shared.
* **Speaker:** Navaneeth Tanuboddi

---

## Slide 8: Error Analysis & Key Insights
* **Challenging Linguistic Patterns:**
  - **Double Negatives:** Posts like *"做事情不要犹豫，要敢做，就像打疫苗，一犹豫就没了"* contain negative tokens ("不要", "犹豫", "没了") but convey strong positive encouragement.
  - **Mixed Ambivalent Sentiment:** *"虽然手臂酸痛得厉害，但为了保护家人也是值得的"* (Negative physical symptom + positive altruistic decision).
  - **Class Boundary Ambiguity:** News posts reporting clinical trials vs subjective personal opinions.
* **Speaker:** Krrish P Raju

---

## Slide 9: Individual Contributions Breakdown
* Evaluated per rubric criteria (10 marks):

| Team Member | SRN | Core Modules & Responsibilities |
| :--- | :--- | :--- |
| **Navaneeth Tanuboddi** | `PES1UG24CS709` | • Preprocessing & Tokenization engine (`preprocessing.py`)<br>• Naive Bayes (Laplace + Interpolation) & SVM pipeline<br>• Temporal Trend Analysis & Figure 1 Dual-Axis Plot<br>• Report & Slide Deck Preparation |
| **Krrish P Raju** | `PES1UG24CS708` | • Evaluation Metrics & Data Splitting (`util.py`)<br>• Regularized Logistic Regression & Bigram Markov Model<br>• Neural LSTM architecture & sequence gating<br>• Interactive Live Demonstration CLI (`demo.py`) |

---

## Slide 10: Live Demonstration Transition
* **Live Demo Commands:**
  ```bash
  python3 demo.py
  python3 run_experiments.py
  ```
* Ready to answer questions from the evaluation panel!
