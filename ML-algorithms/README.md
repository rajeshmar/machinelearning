# 🧠 20 ML Algorithms — From "What Is It" to Running Code

By [Rajesh Kumar Sugumaran](https://github.com/rajeshmar)

This is my working reference for the 20 algorithms a data analyst actually runs into. Not a textbook. Each one answers three questions — what problem does it solve, what's the intuition, and what does the code look like — and then there's a real case where someone used it and it mattered.

I built it while learning this stuff myself, leaning the examples toward supply chain, finance, and retail because that's the work I do. If you're coming from the same direction, it should save you some time.

Everything here runs. I tested the setup on an Apple Silicon Mac (Python 3.13), hit two install snags, and wrote down the fixes so you don't lose an evening to them like I almost did.

---

## 📦 What's inside

The 20 algorithms, grouped by where they sit in real work:

**Beginner**

1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. K-Means Clustering
6. Naive Bayes

**Intermediate**

7. Time Series Forecasting
8. PCA
9. Apriori
10. Gradient Boosting (XGBoost / LightGBM)
11. Recommendation Algorithms
12. Neural Networks

**Advanced**

13. Anomaly Detection
14. DBSCAN
15. A/B Testing
16. NLP
17. Transformers
18. Graph Algorithms
19. Survival Analysis
20. Reinforcement Learning

Each entry follows the same shape: the business problem first, then the intuition in plain English, then runnable Python, a line-by-line walkthrough, a real-world case, and an honest "where it breaks" note. That last part is the one most guides skip and the one that saves you in production.

---

## ⚙️ Setup

You need Python 3.9 or newer. I'd use a virtual environment so you don't pollute your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Then install the stack. This one line covers 19 of the 20 algorithms:

```bash
pip install scikit-learn pandas numpy matplotlib statsmodels xgboost lightgbm mlxtend lifelines
```

Algorithm 17 (Transformers) needs two more libraries. They're large — torch alone is a few hundred MB — so install them only when you actually want that section:

```bash
pip install transformers torch
```

The transformer models themselves download the first time you call `pipeline(...)`, not during install. So if that first run seems to hang, it's pulling a model (a few hundred MB), not frozen. After that it loads from cache instantly.

---

## ✅ Check that it worked

Run this. If both versions print, you're done:

```python
import sklearn, pandas, numpy, statsmodels, xgboost, lightgbm, mlxtend, lifelines
print("all good:", sklearn.__version__, xgboost.__version__)
```

And if you installed the transformer libraries:

```python
import torch, transformers
print("torch:", torch.__version__, "| transformers:", transformers.__version__)
print("Apple GPU available:", torch.backends.mps.is_available())
```

That last line checks for MPS — Apple Silicon's GPU backend. If it says `True`, you can run transformer models on the GPU by passing `device="mps"` to the pipeline. Not required for anything here (the examples are small enough on CPU), but it's a free speedup if you've got it.

---

## 🩹 Two install problems I hit on macOS (and the fixes)

If you're on a Mac, you'll probably trip on these. Here's what happened and what fixed it.

### XGBoost: "Library not loaded"

The import blew up with an `XGBoostError` and a `Library not loaded: libxgboost.dylib` message. The cause: XGBoost's Python package wraps a compiled C++ library that needs OpenMP, and macOS doesn't ship OpenMP. The fix is one line:

```bash
brew install libomp
```

Restart your Python kernel, re-import, done.

One warning if you read the brew output: it says libomp is "keg-only," meaning it wasn't symlinked into the standard path. Usually XGBoost finds it anyway. If it doesn't and you get the same error, symlink it manually:

```bash
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/homebrew/lib/libomp.dylib
```

A heads-up on reading that traceback: XGBoost's error template includes a line saying "You are running 32-bit Python on a 64-bit OS." On Apple Silicon that's almost certainly false — it's a generic list of possible causes, not a diagnosis of your actual problem. The real cause is the `dlopen` message at the very bottom. Don't chase the 32-bit thing.

### Everything else: fine

scikit-learn, lightgbm, statsmodels, and lifelines don't have the OpenMP dependency, so they import without any of this fuss. If you want to confirm the rest of your environment works while you sort out XGBoost, just drop xgboost from the import line.

---

## 🗺️ How to pick an algorithm

Before reaching for anything, three questions narrow it down fast:

1. **Do you have labels** — a known answer column to learn from? Yes → supervised. No → unsupervised.
2. **Is the answer a number or a category?** Number → regression. Category → classification.
3. **Is the data a table, a sequence (time/text), or a network of relationships?**

| Family | Algorithms | Labels? | Output |
|---|---|---|---|
| Regression | Linear, Time Series | Yes | Number |
| Classification | Logistic, Decision Tree, Random Forest, Naive Bayes, XGBoost | Yes | Category |
| Clustering | K-Means, DBSCAN | No | Groups |
| Dimensionality | PCA | No | Compressed features |
| Pattern mining | Apriori | No | Item associations |
| Detection | Anomaly Detection | Usually no | Normal vs. abnormal |
| Recommending | Recommendation | Mixed | Ranked items |
| Deep learning | Neural Nets, Transformers | Yes | Anything |
| Sequential decisions | Reinforcement Learning | Reward signal | Actions |
| Relational | Graph Algorithms | No | Network structure |
| Time-to-event | Survival Analysis | Yes (censored) | Duration probability |
| Experimentation | A/B Testing | — | Statistical decision |
| Language | NLP | Mixed | Text understanding |

---

## 🧭 Where I'd start

If you're learning this to actually use it, not to pass a quiz, the slide-deck order (beginner → advanced) isn't the most useful path. Here's what I'd do instead:

Start with **linear and logistic regression**. They're simple, but more importantly they teach the train/test split and the metrics — precision, recall, AUC, MAE — that every later algorithm reuses. Get those habits right early.

Then jump straight to **XGBoost (#10)**, even though it's labeled intermediate. For tabular business data it's the highest-payoff thing on the whole list. It wins competitions and runs in production fraud and credit systems for a reason.

Add **K-Means and PCA** for the unsupervised side — segmentation and preprocessing, which you'll use constantly.

After that, follow your domain. Mine is supply chain, so #7 (forecasting), #18 (graphs), #19 (survival analysis), and #20 (reinforcement learning for inventory policy) map straight onto problems I recognize. Yours might pull you somewhere else.

The one thread through all of it: start from the business problem, hold out your test data honestly, and check whether a simple model is good enough before you reach for a complicated one. The fancy model is rarely the bottleneck. The data quality and the way you framed the problem usually are.

---

## 📓 A note on the case studies

Some of the real-world examples cite public, reported figures — Google's data-center cooling work, Netflix and recommendations, the Microsoft Bing experiment. Others (the 30-million-subscriber telecom, the 400-turbine operator) are realistic composites built from plausible numbers, there to make the mechanics concrete, not to report a specific company's results. The text flags which is which. If you need every example to be a citable, sourced deployment, that's a different document and I'm happy to build it.

---

## 📄 Author & license

Written and maintained by [Rajesh Kumar Sugumaran](https://github.com/rajeshmar).

MIT. Use it, fork it, fix my mistakes. If you find an error or a better example, open a PR.
