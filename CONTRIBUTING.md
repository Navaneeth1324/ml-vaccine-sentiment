# Contributing Guidelines

Thank you for contributing to the **COVID-19 Vaccine Sentiment Analysis** project!

This repository hosts the machine learning pipeline, dataset generators, model benchmarks, and demonstration utilities for the **UE24CS352A Machine Learning Mini-Project** at PES University.

---

## 👥 Core Maintainers & Owners

- **Navaneeth Tanuboddi** (`PES1UG24CS709`) — GitHub: [@Navaneeth1324](https://github.com/Navaneeth1324)
- **Krrish P Raju** (`PES1UG24CS708`) — GitHub: [@krrish676767](https://github.com/krrish676767)

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Navaneeth1324/ml-vaccine-sentiment.git
   cd ml-vaccine-sentiment
   ```

2. **Configure your Git identity (required for proper contribution attribution):**
   ```bash
   git config user.name "Your Name"
   git config user.email "your-github-registered-email@example.com"
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

---

## 🧪 Running Tests

Before committing code or submitting pull requests, run the test suite to ensure all unit tests pass:

```bash
make test
# or
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📌 Commit Message Conventions

We follow clear semantic commit conventions:
- `feat(...)`: New feature or capability
- `fix(...)`: Bug fix or patch
- `test(...)`: Adding or updating unit tests
- `docs(...)`: Documentation updates
- `ci(...)`: GitHub Actions or automation improvements
- `refactor(...)`: Code refactoring without behavioral changes
