# 20 Algorithms Every Data Analyst Must Know — A Working Engineer's Guide

A practical walkthrough of all 20 algorithms. For each one: the business problem it solves, the intuition, runnable Python, a line-by-line explanation, and a **real-world case study** — a named scenario with actual numbers and the decision the output drove. Examples lean toward supply chain, finance, and retail since that's where most of this lands in real work.

**Setup once.** Everything below assumes:

```bash
pip install scikit-learn pandas numpy matplotlib statsmodels xgboost lightgbm mlxtend lifelines
```

---

## Mental model: where each algorithm fits

Before the details, here's the map. Three questions decide which algorithm you reach for:

1. **Do you have labels?** (a known answer column to learn from) → supervised. No labels → unsupervised.
2. **Is the answer a number or a category?** Number → regression. Category → classification.
3. **Is the data structured (tables), sequential (time/text), or relational (networks)?**

| Family | Algorithms | Labels? | Output |
|---|---|---|---|
| Regression | Linear, Time Series | Yes | Number |
| Classification | Logistic, Decision Tree, Random Forest, Naive Bayes, XGBoost | Yes | Category |
| Clustering | K-Means, DBSCAN | No | Groups |
| Dimensionality | PCA | No | Compressed features |
| Pattern mining | Apriori | No | Item associations |
| Detection | Anomaly Detection | Usually no | Normal vs. abnormal |
| Ranking/Suggesting | Recommendation | Mixed | Ranked items |
| Deep learning | Neural Networks, Transformers | Yes | Anything |
| Sequential decisions | Reinforcement Learning | Reward signal | Actions |
| Relational | Graph Algorithms | No | Network structure |
| Time-to-event | Survival Analysis | Yes (censored) | Duration probability |
| Experimentation | A/B Testing | — | Statistical decision |
| Language | NLP | Mixed | Text understanding |

---

# BEGINNER LEVEL

## 1. Linear Regression

**Business problem.** You want to predict a continuous number and you believe inputs move the output in a roughly straight-line way. "If I spend X on ads, what revenue do I get?" "Given lead time and order volume, what's my expected inventory cost?"

**Intuition.** Fit the straight line (or flat plane in higher dimensions) that sits as close as possible to all your data points. "Close" means minimizing the sum of squared vertical distances between the line and each point. The model learns a slope for each input and one intercept. The slope tells you: for every one-unit increase in this input, the output changes by this much.

The equation is just `y = β₀ + β₁x₁ + β₂x₂ + ... + error`. That's it. Everything else is bookkeeping.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Simulated demand data: ad spend + price -> weekly units sold
rng = np.random.default_rng(42)
n = 500
ad_spend = rng.uniform(1000, 10000, n)
price = rng.uniform(20, 80, n)
# true relationship + noise
units = 50 + 0.02 * ad_spend - 1.5 * price + rng.normal(0, 30, n)

df = pd.DataFrame({"ad_spend": ad_spend, "price": price, "units": units})
X = df[["ad_spend", "price"]]
y = df["units"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Coefficients:", dict(zip(X.columns, model.coef_.round(4))))
print("Intercept:", round(model.intercept_, 2))
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R²:", round(r2_score(y_test, pred), 3))
```

**Walkthrough.**
- `train_test_split` holds back 20% of data the model never sees during training, so the score reflects real-world performance, not memorization.
- `model.fit` finds the coefficients that minimize squared error. Behind the scenes it's solving a linear algebra equation (no iteration needed for ordinary least squares).
- `model.coef_` gives you the slopes. Here ad_spend's coefficient near 0.02 means each extra dollar of ad spend adds ~0.02 units; price's negative coefficient means raising price reduces units. This interpretability is linear regression's superpower.
- **MAE** = average absolute prediction error in real units (easy to explain to a manager). **R²** = fraction of variance explained, 0 to 1; closer to 1 is better.

**Real-world case study — Walmart weather-driven demand.** Retailers have long used linear regression to quantify how outside factors move sales. A classic, well-documented example: ahead of forecast hurricanes, analysis of historical sales found that strawberry Pop-Tarts sales spiked roughly 7× their normal rate in affected stores, and the top-selling item before a hurricane was beer. A regression of unit sales on variables like a storm-warning flag, store region, and day-of-week lets a planner say "a category-3 warning adds N units of demand to these SKUs in this region" — and pre-position stock accordingly. The value isn't a fancy model; it's the *coefficient* turning a vague hunch ("storms change buying") into a stockable number.

**Where it breaks.** If the true relationship is curved, a straight line underfits. Outliers drag the line because errors are *squared*. And correlated inputs (multicollinearity) make individual coefficients unstable even if predictions stay fine.

**Business problem.** You want a yes/no answer *with a probability attached*. "Will this customer churn?" "Is this transaction fraud?" "Will this PO ship late?" You don't just want the label — you want "73% likely to churn" so you can rank and prioritize.

**Intuition.** Start with linear regression's straight-line score, then squash it through an S-shaped curve (the sigmoid) that maps any number to a probability between 0 and 1. Big positive score → near 1. Big negative → near 0. Around zero → near 0.5. You then pick a threshold (often 0.5) to convert probability into a decision.

Despite the name, it's a **classification** algorithm, not regression.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

rng = np.random.default_rng(7)
n = 800
tenure = rng.uniform(1, 60, n)          # months as customer
monthly_charge = rng.uniform(20, 120, n)
support_tickets = rng.poisson(2, n)
# churn more likely with low tenure, high charges, many tickets
logit = -2 + (-0.05 * tenure) + (0.03 * monthly_charge) + (0.4 * support_tickets)
prob = 1 / (1 + np.exp(-logit))
churn = rng.binomial(1, prob)

df = pd.DataFrame({"tenure": tenure, "monthly_charge": monthly_charge,
                   "support_tickets": support_tickets, "churn": churn})
X = df.drop(columns="churn")
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]   # probability of churn
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
```

**Walkthrough.**
- `predict_proba(...)[:, 1]` returns the probability of the positive class (churn=1). This is what you actually use in business — sort customers by churn probability and target the top slice.
- `classification_report` gives **precision** (of those flagged, how many truly churned) and **recall** (of all churners, how many we caught). These trade off against each other and matter more than raw accuracy when classes are imbalanced.
- **ROC-AUC** measures how well the model ranks positives above negatives across all thresholds. 0.5 = random guessing, 1.0 = perfect. It's threshold-independent, which is why it's the go-to single metric.

**Real-world case study — telecom churn at scale.** A mobile carrier with 30 million subscribers and ~2% monthly churn loses roughly 600,000 customers a month. Acquiring a replacement costs 5–7× more than retaining one. The retention team can't call everyone, so they fit logistic regression on tenure, contract type, support-ticket count, and recent data-speed complaints, then rank all subscribers by churn probability. They target the top 5% (highest-risk) with a retention offer. Because the model outputs a *probability*, not just a label, they can tune the cutoff to match the call-center's capacity — extend the campaign to the top 8% in a slow month, tighten to the top 3% when staff is stretched. A modest 10% reduction in churn among the targeted group, on a base that size, is tens of thousands of saved customers a month.

**Where it breaks.** Assumes a roughly linear relationship between inputs and the *log-odds*. Curved decision boundaries need tree-based models or feature engineering.

---

## 3. Decision Tree

**Business problem.** You need a model a non-technical stakeholder can actually read and trust. "Why was this loan rejected?" A tree answers with an if-then chain: "Income below 30k AND no collateral → reject." Used heavily in risk, healthcare, and HR where explainability is a legal or trust requirement.

**Intuition.** Repeatedly split the data on the question that best separates outcomes. At each node the tree asks "which single yes/no question reduces the most disorder in my groups?" It keeps splitting until groups are pure enough or you hit a stopping rule. To predict, you walk a new example down the branches to a leaf.

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(1)
n = 600
income = rng.uniform(15000, 120000, n)
debt_ratio = rng.uniform(0, 1, n)
years_employed = rng.uniform(0, 20, n)
# approve if decent income, low debt ratio
approve = ((income > 40000) & (debt_ratio < 0.5) | (years_employed > 10)).astype(int)

df = pd.DataFrame({"income": income, "debt_ratio": debt_ratio,
                   "years_employed": years_employed, "approve": approve})
X = df.drop(columns="approve")
y = df["approve"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# max_depth limits complexity -> prevents memorizing noise
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy:", round(accuracy_score(y_test, tree.predict(X_test)), 3))
print("\nThe actual decision rules:\n")
print(export_text(tree, feature_names=list(X.columns)))
```

**Walkthrough.**
- `max_depth=3` is the single most important knob. Without it, a tree grows until every leaf is one data point — perfect on training data, useless on new data. That's overfitting in its purest form.
- `export_text` prints the literal rules. This is the whole appeal: you can hand these to a compliance officer.
- The split criterion (Gini impurity by default) measures how mixed each group is. The tree greedily picks splits that produce the purest children.

**Real-world case study — hospital triage explainability.** A hospital wants to flag emergency-department patients at high risk of deterioration in the next 24 hours. A black-box model that's slightly more accurate is a non-starter: a nurse won't act on "the computer says so," and the hospital's review board needs to audit every decision. A decision tree trained on vitals (heart rate, blood pressure, oxygen saturation, age) produces rules a clinician reads in seconds — "oxygen saturation below 92% AND respiratory rate above 24 → high risk." The interpretability *is* the deliverable. Even where a more accurate model exists, the tree often gets deployed as the human-facing layer because every flag comes with a reason the staff can verify against the patient in front of them.

**Where it breaks.** A single tree is unstable — change a few data points and the whole structure can shift. It also tends to overfit. The fix is to combine many trees, which is exactly what the next two algorithms do.

---

## 4. Random Forest

**Business problem.** You want decision-tree-level flexibility but you need it to be reliable and accurate enough to deploy. Fraud detection, credit scoring, customer churn — anywhere a single tree is too jumpy.

**Intuition.** Build hundreds of decision trees, each on a random subset of the data *and* a random subset of features, then average their votes. Any single tree makes mistakes, but the mistakes are uncorrelated, so they cancel out in the average. This is the "wisdom of crowds" applied to models. The technical name is *bagging* (bootstrap aggregating).

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(3)
n = 1000
amount = rng.exponential(200, n)
hour = rng.integers(0, 24, n)
foreign = rng.binomial(1, 0.1, n)
velocity = rng.poisson(1, n)            # txns in last hour
# fraud: large amounts at odd hours, foreign, high velocity
score = (amount > 500).astype(int) + (hour < 5).astype(int) + foreign + (velocity > 3).astype(int)
fraud = (score >= 2).astype(int)

df = pd.DataFrame({"amount": amount, "hour": hour, "foreign": foreign,
                   "velocity": velocity, "fraud": fraud})
X = df.drop(columns="fraud")
y = df["fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)

print("Accuracy:", round(accuracy_score(y_test, rf.predict(X_test)), 3))
print("\nFeature importance (what drives fraud):")
for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda t: -t[1]):
    print(f"  {feat:12s} {imp:.3f}")
```

**Walkthrough.**
- `n_estimators=200` means 200 trees. More trees = more stable, with diminishing returns and more compute. 100–300 is a typical sweet spot.
- `n_jobs=-1` uses all CPU cores — the trees are independent so they train in parallel for free.
- `feature_importances_` ranks which inputs the forest relied on most. This is your "what actually matters" report for stakeholders, though note it can be biased toward high-cardinality features.

**Real-world case study — credit scoring at a lender.** An online lender approves or declines loan applications in seconds. A single decision tree is too jumpy — two near-identical applicants can land on opposite sides of a split and get opposite decisions, which is both bad business and a fair-lending risk. A random forest of a few hundred trees smooths this: the decision is a vote, so a borderline applicant gets a stable, reproducible score. Just as useful is the `feature_importances_` output — when it shows that "number of recent credit inquiries" and "debt-to-income ratio" dominate, the risk team can explain to regulators what drives decisions and check that no prohibited feature (like a zip-code proxy for race) is sneaking in as a top driver. Stability plus an auditable importance ranking is why forests remain a workhorse in credit.

**Where it breaks.** Slower and larger than one tree. Less interpretable (you can't print 200 trees as rules). For maximum accuracy on structured data, boosting (next) usually edges it out.

---

## 5. K-Means Clustering

**Business problem.** No labels, but you suspect natural groups exist. "Segment my customers so marketing can target each group differently." "Group SKUs by demand pattern." You don't know the groups ahead of time — you want the algorithm to find them.

**Intuition.** You decide there are *k* groups. The algorithm places *k* center points, assigns each data point to its nearest center, then moves each center to the average of its assigned points. Repeat until centers stop moving. The result: points within a cluster are close to each other and far from other clusters.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(11)
n = 500
# three latent customer types
recency = np.concatenate([rng.normal(10, 3, n), rng.normal(40, 5, n), rng.normal(80, 8, n)])
frequency = np.concatenate([rng.normal(20, 4, n), rng.normal(8, 2, n), rng.normal(2, 1, n)])
monetary = np.concatenate([rng.normal(500, 80, n), rng.normal(200, 40, n), rng.normal(50, 15, n)])

df = pd.DataFrame({"recency": recency, "frequency": frequency, "monetary": monetary})

# CRITICAL: scale first, or 'monetary' dominates just by being a bigger number
X = StandardScaler().fit_transform(df)

km = KMeans(n_clusters=3, random_state=0, n_init=10)
df["segment"] = km.fit_predict(X)

print(df.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1))
```

**Walkthrough.**
- **Scaling is mandatory.** K-Means uses distance. If monetary runs 0–500 and frequency runs 0–20, the algorithm thinks monetary matters 25× more purely because of units. `StandardScaler` puts everything on the same footing.
- `n_init=10` runs the algorithm 10 times with different random starts and keeps the best. K-Means can land in a bad local solution depending on where centers start, so this guards against it.
- Choosing *k* is the hard part. Use the **elbow method** (plot inertia vs. k, look for the bend) or **silhouette score**. Here we cheated by knowing there are 3 groups.

**Real-world case study — RFM segmentation for an e-commerce retailer.** A retailer with 200,000 customers wants marketing to stop blasting the same email to everyone. They compute three numbers per customer — Recency (days since last order), Frequency (orders in the last year), Monetary (total spend) — and run K-Means with k=4. Out come four readable segments: "champions" (recent, frequent, high spend), "loyal but cooling" (frequent historically, slipping on recency), "big-ticket occasionals" (rare orders, high value), and "at-risk" (haven't bought in months). Each segment gets a different play — champions get early access and referrals, at-risk customers get a win-back discount, big-ticket occasionals get high-margin product launches. The same email budget, redirected by segment, typically lifts campaign conversion meaningfully versus one-size-fits-all because the message finally matches the customer. This RFM-plus-K-Means recipe is one of the most widely deployed clustering applications in retail.

**Where it breaks.** Assumes clusters are roughly spherical and similar in size. It forces every point into a cluster (no concept of noise) and you must pick *k* upfront. When those assumptions fail, DBSCAN (algorithm 14) handles it.

---

## 6. Naive Bayes

**Business problem.** Fast text classification at scale. Spam filtering, routing support tickets, sentiment tagging on reviews. When you have thousands of word-features and need a model that trains in milliseconds and predicts instantly.

**Intuition.** Apply Bayes' theorem to update the probability of each class given the evidence (the words present). The "naive" part: it assumes every feature is independent of the others given the class — that the word "free" and the word "winner" appearing in spam are unrelated. This is almost never literally true, but the model works shockingly well anyway because it only needs to rank classes correctly, not estimate exact probabilities.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "win free money now", "free prize claim today", "urgent win cash prize",
    "meeting scheduled for monday", "please review the attached report",
    "lunch tomorrow at noon", "claim your free reward instantly",
    "project update and timeline", "team sync this afternoon",
    "congratulations you won a lottery"
]
labels = ["spam", "spam", "spam", "ham", "ham", "ham", "spam", "ham", "ham", "spam"]

# CountVectorizer turns text into word-count features; NB classifies
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

tests = ["free cash prize win now", "can we schedule a meeting tomorrow"]
for t, p in zip(tests, model.predict(tests)):
    print(f"'{t}' -> {p}")
```

**Walkthrough.**
- `CountVectorizer` builds a vocabulary from all words seen and converts each message into a vector of word counts. "win free money now" becomes counts across the vocabulary.
- `MultinomialNB` is the variant for word counts. It learns P(word | spam) and P(word | ham) from training data, then for a new message multiplies these together (the naive independence step) to score each class.
- `make_pipeline` chains the two so the same text transformation applies to training and prediction automatically — no leakage, no manual steps.

**Real-world case study — email spam filtering.** Naive Bayes is the algorithm behind the spam filters that defined early webmail. The reason it won that job: an inbox provider processes billions of messages a day, so the classifier has to score each one in microseconds and retrain cheaply as spammers shift tactics. Naive Bayes learns P(word | spam) for tens of thousands of words and multiplies them — arithmetic so fast it runs at email-firehose scale on modest hardware. When spammers started misspelling "viagra" as "v1agra," retraining on fresh labeled mail updated the word probabilities overnight. It's rarely the *most* accurate option today, but the combination of speed, tiny memory footprint, and trivial retraining kept it in production filters for years and still makes it the default first pass for high-volume text routing.

**Where it breaks.** The independence assumption hurts when features are strongly correlated. It gives poor *probability estimates* (often overconfident) even when the *classification* is right — so trust the predicted class, not the raw probability.

---

# INTERMEDIATE LEVEL

## 7. Time Series Forecasting

**Business problem.** Predict future values when your data has a time order and the past predicts the future: demand planning, revenue forecasting, inventory levels, staffing. This is daily bread in supply chain.

**Intuition.** Decompose the series into **trend** (long-term direction), **seasonality** (repeating cycles — weekly, monthly, yearly), and **noise**. A good forecaster models the structure and projects it forward. Classic approach: ARIMA (autoregression + differencing + moving average). Modern shops often use Prophet or gradient-boosted models with lag features, but ARIMA teaches the fundamentals.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Synthetic monthly demand: upward trend + 12-month seasonality + noise
rng = np.random.default_rng(5)
months = pd.date_range("2021-01-01", periods=48, freq="MS")
trend = np.linspace(100, 200, 48)
seasonal = 20 * np.sin(np.arange(48) * 2 * np.pi / 12)
demand = trend + seasonal + rng.normal(0, 8, 48)
ts = pd.Series(demand, index=months)

train, test = ts[:42], ts[42:]

# ARIMA(p,d,q): p=past values, d=differencing for trend, q=past errors
model = ARIMA(train, order=(2, 1, 2)).fit()
forecast = model.forecast(steps=6)

mae = np.mean(np.abs(forecast.values - test.values))
print("Forecast vs actual (next 6 months):")
for date, f, a in zip(test.index, forecast.values, test.values):
    print(f"  {date:%Y-%m}  forecast={f:6.1f}  actual={a:6.1f}")
print(f"\nMAE: {mae:.2f}")
```

**Walkthrough.**
- `order=(2,1,2)`: the AR term (2) uses the last 2 values, the I term (1) differences once to remove trend, the MA term (2) corrects using the last 2 forecast errors. Tuning these is the craft; tools like `pmdarima.auto_arima` automate the search.
- `forecast(steps=6)` projects 6 months ahead. Note forecasts get less certain the further out you go.
- **Critical discipline:** never shuffle time series data into a random train/test split. The test set must come *after* the training set in time, or you leak the future into the past.

**Real-world case study — supply chain demand planning.** A consumer-goods company stocking 5,000 SKUs across 20 distribution centers lives or dies on the demand forecast. Forecast too low and you stock out — lost sales plus emergency air-freight to refill. Forecast too high and cash sits frozen in inventory plus warehousing and obsolescence cost. The planning team forecasts each SKU-DC combination monthly, capturing trend (a product ramping or declining) and seasonality (sunscreen peaks in summer, the obvious one; less obvious, B2B orders dipping every December). A forecast accuracy improvement of even a few percentage points of MAPE, multiplied across 100,000 SKU-DC combinations, translates directly into millions in freed working capital and fewer stockouts. This is the single most common time-series application in operations, and it's exactly the kind of problem that sits at the center of supply-chain analytics work.

**Where it breaks.** ARIMA assumes the statistical structure stays stable. Sudden regime changes (a pandemic, a price war) break it. Always pair forecasts with confidence intervals and human judgment.

---

## 8. PCA (Principal Component Analysis)

**Business problem.** Too many features. You have 200 columns, many correlated, and your models are slow, overfitting, or impossible to visualize. PCA compresses them into a handful of new features that keep most of the information.

**Intuition.** Find the directions in your data where the points spread out the most (the directions of maximum variance). The first principal component is the single line that captures the most spread; the second captures the most of what's left while being perpendicular to the first; and so on. Project your data onto the top few components and you've compressed 200 dimensions into, say, 10 with minimal information loss.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data            # 569 samples, 30 features
print("Original shape:", X.shape)

X_scaled = StandardScaler().fit_transform(X)   # scale first, always

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)
print("Reduced shape:", X_reduced.shape)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print("\nVariance retained by first N components:")
for i, v in enumerate(cumvar, 1):
    print(f"  {i:2d} components -> {v:.1%}")
```

**Walkthrough.**
- **Scale first.** Like K-Means, PCA is variance-based, so a feature measured in thousands would dominate purely by scale.
- `explained_variance_ratio_` tells you how much information each component keeps. The cumulative sum is the headline number: if 10 components retain 95% of variance, you've cut features by two-thirds and lost almost nothing.
- The new components are combinations of original features, so they lose direct interpretability — "PC1" doesn't map to a real-world quantity. That's the trade-off for compression.

**Real-world case study — sensor compression in predictive maintenance.** A factory instruments a production line with 120 sensors (vibration, temperature, pressure, current draw) sampling every second. Many of these readings move together — when a bearing heats up, nearby temperature *and* vibration sensors all rise. Feeding 120 correlated signals straight into a fault-detection model is slow and noisy. PCA compresses them to ~8 components that retain 95% of the variance, and a striking byproduct emerges: the first few components often correspond to *physically meaningful* modes of the machine (one tracks overall load, another tracks an imbalance signature). The maintenance team monitors those few components instead of 120 raw dials, and a sudden jump in a normally-stable component becomes an early warning of a developing fault. Same information, a fraction of the dimensions, and a cleaner signal to alarm on.

**Where it breaks.** PCA only captures *linear* structure and assumes variance equals importance, which isn't always true (a low-variance feature might be the one that predicts your target). Use it for preprocessing and visualization, not as a substitute for thinking about which features matter.

---

## 9. Apriori Algorithm

**Business problem.** "What products get bought together?" Market basket analysis for cross-selling, store layout, bundle pricing, and recommendation seeds. The classic "customers who bought X also bought Y."

**Intuition.** Find sets of items that frequently appear together in transactions, then turn them into rules. The key trick (the "Apriori property"): if a set of items is rare, any larger set containing it is also rare — so you can prune huge swaths of combinations without checking them. Three metrics matter: **support** (how often a combo appears), **confidence** (given X, how often Y follows), and **lift** (how much more likely Y is when X is present vs. baseline).

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["bread", "milk", "eggs"],
    ["bread", "butter"],
    ["milk", "butter", "eggs"],
    ["bread", "milk", "butter"],
    ["bread", "milk", "butter", "eggs"],
    ["milk", "eggs"],
    ["bread", "butter"],
]

te = TransactionEncoder()
arr = te.fit_transform(transactions)            # one-hot: item present per basket
df = pd.DataFrame(arr, columns=te.columns_)

freq = apriori(df, min_support=0.3, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)

cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(rules[cols].sort_values("lift", ascending=False).round(3).to_string(index=False))
```

**Walkthrough.**
- `TransactionEncoder` turns each basket into a row of True/False flags (one column per product). This is the format Apriori needs.
- `min_support=0.3` means we only consider item sets appearing in at least 30% of baskets — this prunes noise and controls compute.
- In the rules table, **lift > 1** means the items appear together more than chance would predict — that's a real association worth acting on. **Confidence** tells you the strength: confidence 0.8 for {bread} → {butter} means 80% of bread buyers also buy butter.

**Real-world case study — grocery store layout and bundle pricing.** A supermarket chain runs market-basket analysis on a month of loyalty-card transactions and finds a strong rule: {diapers} → {baby wipes} with high confidence and lift well above 1, plus a weaker but real {pasta} → {pasta sauce}. They act on it three ways. First, layout — place wipes near diapers so the obvious pairing is frictionless, but place high-margin items *along the path* between two staples that customers always buy together, capturing impulse buys on the walk. Second, bundle pricing — a "pasta night" bundle priced just below buying the items separately lifts units on both. Third, the recommendation seed — "frequently bought together" on the e-commerce site is literally these rules. The famous (and partly apocryphal) "beer and diapers" story comes from exactly this kind of analysis; the durable lesson is that the rules surface pairings no merchandiser would have guessed, which is the whole value.

**Where it breaks.** Combinatorially expensive with large catalogs. FP-Growth is a faster alternative. Also, correlation isn't causation — a lift doesn't mean placing items together *causes* more sales; test it with an experiment (see A/B Testing).

---

## 10. Gradient Boosting (XGBoost / LightGBM)

**Business problem.** Maximum predictive accuracy on structured/tabular data. This is the algorithm that wins most Kaggle competitions on tables and powers a huge share of production credit-scoring, fraud, ranking, and demand models. If your data is in rows and columns and you want the best number, start here.

**Intuition.** Build trees *sequentially*, where each new tree focuses on the errors the previous trees made. Tree 1 makes a rough prediction; tree 2 learns to predict tree 1's mistakes; tree 3 corrects what's left; and so on. Each tree is weak alone, but adding hundreds of error-correcting trees produces a very strong model. Contrast with Random Forest, which builds trees *independently and in parallel*; boosting builds them *dependently and in sequence*.

```python
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=3000, n_features=20, n_informative=10,
                           weights=[0.85, 0.15], random_state=0)  # imbalanced like fraud
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = XGBClassifier(
    n_estimators=300,        # number of sequential trees
    learning_rate=0.05,      # how much each tree corrects (smaller = safer, slower)
    max_depth=4,             # shallow trees to avoid overfitting
    subsample=0.8,           # row sampling per tree -> regularization
    colsample_bytree=0.8,    # feature sampling per tree
    eval_metric="auc",
    random_state=0,
)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("ROC-AUC:", round(auc, 3))
```

**Walkthrough.**
- `learning_rate` and `n_estimators` work as a pair: a smaller learning rate needs more trees but generalizes better. The classic recipe is low learning rate (0.01–0.1) + many trees + early stopping.
- `subsample` and `colsample_bytree` add randomness so trees don't all chase the same patterns — this is regularization that fights overfitting.
- `max_depth=4`: boosting uses *shallow* trees deliberately. Each is a weak learner; depth comes from stacking many, not from any single deep tree.
- **LightGBM** is the faster sibling — same idea, smarter tree-growth strategy, better on large datasets. Swap `XGBClassifier` for `lightgbm.LGBMClassifier` with near-identical params.

**Real-world case study — real-time payment fraud.** A payments processor scores every card transaction in under 100 milliseconds to approve or block it. The data is tabular (amount, merchant category, time since last transaction, distance from last location, device fingerprint) and the fraud rate is well under 1% — a brutally imbalanced problem where accuracy is meaningless (a model predicting "never fraud" is 99%+ accurate and useless). Gradient boosting handles both the tabular structure and the imbalance better than almost anything else, which is why XGBoost and LightGBM dominate production fraud systems. The business tuning is all about the precision/recall trade-off in dollars: each false positive is a declined legitimate customer (annoyed, maybe churns), each false negative is a chargeback the processor eats. The team picks the score threshold where the marginal cost of one more block equals the marginal fraud caught — and re-tunes it as fraud patterns shift. This is the canonical "best model for tabular data" use case.

**Where it breaks.** More hyperparameters to tune than Random Forest, and easier to overfit if you're careless (high learning rate + deep trees + too many estimators). Use a validation set with early stopping. On unstructured data (images, raw text) deep learning wins instead.

---

## 11. Recommendation Algorithms

**Business problem.** "What should we show this user next?" Netflix titles, Amazon products, Spotify tracks, YouTube videos. The goal is personalization that increases engagement and sales.

**Intuition.** Two main families. **Collaborative filtering**: "people similar to you liked this" — it finds patterns in who-liked-what without knowing anything about the items themselves. **Content-based**: "this is similar to things you already liked" — it uses item attributes. The example below is item-based collaborative filtering: find items that tend to be rated similarly by the same users.

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# user-item rating matrix (rows=users, cols=movies, 0 = not rated)
ratings = pd.DataFrame({
    "Matrix":     [5, 4, 0, 1, 0],
    "Inception":  [5, 0, 0, 1, 2],
    "Titanic":    [1, 1, 5, 4, 0],
    "Notebook":   [0, 1, 5, 5, 4],
    "Avengers":   [4, 5, 1, 0, 0],
}, index=["Alice", "Bob", "Carol", "Dave", "Eve"])

# item-item similarity: how similarly are two movies rated across users?
item_sim = pd.DataFrame(
    cosine_similarity(ratings.T),
    index=ratings.columns, columns=ratings.columns
)

def recommend(user, n=2):
    seen = ratings.loc[user]
    scores = item_sim.dot(seen).div(item_sim.sum(axis=1))  # weighted by similarity
    scores = scores[seen == 0]                              # drop already-rated
    return scores.sort_values(ascending=False).head(n)

print("Recommendations for Alice:")
print(recommend("Alice").round(3))
```

**Walkthrough.**
- `cosine_similarity(ratings.T)` measures the angle between item rating-vectors. Two movies rated highly by the same users point in the same direction → high similarity. Cosine ignores magnitude, so it handles users who rate generously vs. harshly.
- The `recommend` function scores unseen movies by blending similar movies' ratings, weighted by how similar they are, then filters out what the user already saw.
- Real systems (matrix factorization, deep learning two-tower models) scale this to millions of users, handle the "cold start" problem for new users/items, and mix in content features. The intuition stays the same.

**Real-world case study — Netflix and the watch-next problem.** Netflix has reported that a large majority of what members watch comes from its recommendations rather than search — the rows of "because you watched X" and the personalized artwork are the product, not a feature bolted on. The business stakes are concrete: the company has publicly framed personalized recommendations as worth on the order of a billion dollars a year, mostly through reduced churn — every member who finds something to watch tonight is a member who doesn't cancel this month. The system blends collaborative filtering (members with similar taste liked this) with content signals and context (time of day, device). The cold-start problem shows up the moment you sign up — which is why Netflix asks new members to pick a few titles they like, bootstrapping the very first recommendations before any watch history exists.

**Where it breaks.** Cold start — you can't recommend to a brand-new user with no history. Popularity bias — popular items get over-recommended, creating filter bubbles. Sparsity — most users rate almost nothing.

---

## 12. Neural Networks

**Business problem.** Learn complex, nonlinear patterns from large data where you can't hand-engineer the rules: image recognition, language, speech, anything with rich structure. The foundation under modern AI.

**Intuition.** Stack layers of simple units ("neurons"). Each neuron takes a weighted sum of its inputs, passes it through a nonlinear activation function, and sends the result forward. Stacking nonlinear layers lets the network approximate almost any function. Training works by **backpropagation**: make a prediction, measure the error, then push that error backward through the layers adjusting every weight a little to reduce it. Repeat millions of times.

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 'moons' = two interleaving crescents; impossible for a straight-line model
X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# two hidden layers of 16 and 8 neurons
nn = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                   max_iter=2000, random_state=0)
nn.fit(X_train, y_train)

print("Accuracy:", round(accuracy_score(y_test, nn.predict(X_test)), 3))
print("Layers (in, hidden..., out):", [c.shape for c in nn.coefs_])
```

**Walkthrough.**
- `hidden_layer_sizes=(16, 8)` defines the architecture: input → 16 neurons → 8 neurons → output. More/wider layers = more capacity to learn complex patterns, but more risk of overfitting and more data/compute needed.
- `activation="relu"` is the nonlinearity (outputs the input if positive, else 0). Without a nonlinear activation, stacking layers would collapse into a single linear model — the nonlinearity is what gives networks their power.
- The moons dataset is the point: a linear model can't separate two interleaved crescents, but a small network draws the curved boundary easily.
- Real deep learning uses PyTorch or TensorFlow for GPU training, custom architectures, and far larger scale. `MLPClassifier` is the conceptual starting point.

**Real-world case study — diabetic retinopathy screening.** In many regions there aren't enough ophthalmologists to screen every diabetic patient for retinopathy, a leading cause of preventable blindness that's treatable if caught early. A convolutional neural network trained on tens of thousands of retinal photographs learned to grade disease severity from the image alone, reaching specialist-level agreement in published studies and getting deployed in screening clinics where a technician captures the photo and the model flags who needs to see a specialist. The point for understanding neural nets: nobody hand-coded "look for these microaneurysms." The network learned the visual features directly from labeled images — exactly the kind of rich, nonlinear pattern that's impossible to specify as rules but learnable from enough examples. That's the capability that separates neural networks from everything above them in this guide.

**Where it breaks.** Data-hungry and compute-hungry. A black box — hard to explain *why* it predicted something. For small tabular datasets, gradient boosting usually beats a neural net with far less fuss.

---

## 13. Anomaly Detection

**Business problem.** Catch the rare, suspicious, or broken thing automatically: fraud, network intrusions, failing machines on a production line, sensor faults. Often you don't have labeled examples of "bad" — you only know what "normal" looks like.

**Intuition.** Learn the shape of normal data, then flag anything that doesn't fit. Isolation Forest takes a clever angle: anomalies are *easy to isolate*. If you randomly split the data with random cuts, an outlier gets separated from everything else in just a few cuts, while a normal point buried in a dense region takes many cuts. Fewer cuts to isolate = more anomalous.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(9)
normal = rng.normal(0, 1, (480, 2))             # dense normal cluster
anomalies = rng.uniform(-6, 6, (20, 2))         # scattered outliers
X = np.vstack([normal, anomalies])

# contamination = expected fraction of anomalies
iso = IsolationForest(contamination=0.04, random_state=0)
pred = iso.fit_predict(X)        # -1 = anomaly, 1 = normal
scores = iso.score_samples(X)    # lower = more anomalous

n_flagged = (pred == -1).sum()
print(f"Flagged {n_flagged} anomalies out of {len(X)} points")
print("5 most anomalous point scores:", np.sort(scores)[:5].round(3))
```

**Walkthrough.**
- `contamination` is your prior estimate of how much of the data is anomalous. It sets the threshold. Set it from domain knowledge (fraud rate, defect rate); getting it wrong shifts how aggressively the model flags.
- `fit_predict` returns -1/1 labels; `score_samples` gives a continuous anomaly score so you can *rank* by severity and investigate the worst first — usually more useful than a hard flag.
- Isolation Forest is unsupervised: it needs no labeled anomalies, which is the whole point since anomalies are by definition rare and hard to collect.

**Real-world case study — manufacturing quality control.** A bottling line fills 1,200 units a minute. Defects — underfill, a missing cap, a label skew — are rare, maybe a fraction of a percent, and you have almost no labeled examples of *each specific* failure mode because new ones appear all the time. You can't train a supervised classifier on defect types you haven't seen yet. So the line streams sensor and vision features into an Isolation Forest trained only on "normal" runs; anything that doesn't fit the normal envelope gets pulled for inspection. The `contamination` parameter is set from the historical scrap rate, and the *ranked* anomaly score matters more than the binary flag — the QC operator works the worst offenders first when the reject bin fills faster than they can check everything. The unsupervised framing is the whole reason it works: it catches novel defects no labeled dataset would have covered.

**Where it breaks.** Defining "normal" is tricky when normal itself drifts over time (concept drift) — a model trained last year may flag this year's legitimate behavior. And a clever adversary (fraudster) actively tries to look normal.

---

## 14. DBSCAN

**Business problem.** Clustering when groups are oddly shaped, you don't know how many there are, and you have noise/outliers you want excluded rather than forced into a group. Geospatial clustering (delivery hotspots), outlier detection, irregular customer segments.

**Intuition.** Density-based. A cluster is a dense region of points. Pick two settings: `eps` (how close counts as "neighbor") and `min_samples` (how many neighbors make a point a "core" point). The algorithm grows clusters outward from core points through their dense neighborhoods. Points in no dense region are labeled noise — DBSCAN is allowed to say "this point belongs to nothing," which K-Means cannot.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# two crescents + scattered noise: K-Means fails here, DBSCAN shines
X, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
rng = np.random.default_rng(0)
noise = rng.uniform(-2, 3, (20, 2))
X = StandardScaler().fit_transform(np.vstack([X, noise]))

db = DBSCAN(eps=0.25, min_samples=5)
labels = db.fit_predict(X)        # -1 = noise

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points:   {n_noise}")
```

**Walkthrough.**
- `eps=0.25` is the neighborhood radius and the most sensitive knob. Too small → everything is noise; too large → everything merges into one blob. Scale your data first so `eps` means the same thing in every direction.
- `min_samples` controls how dense a region must be to seed a cluster. Higher values → more points labeled noise.
- The headline advantage: DBSCAN found the right number of clusters *automatically* and isolated the noise — no need to specify *k* like K-Means, and crescent shapes that break K-Means work fine here.

**Real-world case study — ride-hailing demand hotspots.** A ride-hailing operator wants to reposition idle drivers toward where riders will request trips. They have millions of GPS pickup points across a city. K-Means is the wrong tool here — it would force every point into a round cluster and pick an arbitrary k, splitting one long demand corridor (a nightlife strip) into chunks while merging unrelated areas. DBSCAN instead finds dense pickup regions of *any shape* — it traces the actual contour of a stadium exit, an airport terminal, a bar district — and labels sparse suburban points as noise rather than inventing a cluster for them. The operator gets a map of real, irregularly-shaped demand zones to stage drivers in, and the count of zones emerges from the data instead of being guessed. Density-based clustering on spatial coordinates is one of DBSCAN's most natural fits.

**Where it breaks.** Struggles when clusters have very different densities (one `eps` can't fit all). Tuning `eps` and `min_samples` takes iteration. High dimensions weaken the density notion (the "curse of dimensionality").

---

# ADVANCED LEVEL

## 15. A/B Testing

**Business problem.** "Did this change actually help, or did we get lucky?" New checkout button, pricing page, email subject line, feature rollout. You need a statistically defensible yes/no before betting the business on it.

**Intuition.** Randomly split users into group A (control, old version) and group B (treatment, new version). Measure the metric you care about (conversion, revenue per user) in each. Then ask: is the observed difference larger than what random noise could plausibly produce? A statistical test gives you a **p-value** — the probability of seeing a difference this big if the two versions were truly identical. Small p-value → the difference is likely real.

```python
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# Control: 1000 users, 100 converted (10%). Treatment: 1000 users, 130 converted (13%).
conversions = np.array([100, 130])
visitors = np.array([1000, 1000])

stat, pval = proportions_ztest(conversions, visitors)
rate_a, rate_b = conversions / visitors
lift = (rate_b - rate_a) / rate_a

print(f"Control conversion:   {rate_a:.1%}")
print(f"Treatment conversion: {rate_b:.1%}")
print(f"Relative lift:        {lift:+.1%}")
print(f"p-value:              {pval:.4f}")
print("Verdict:", "Significant — ship it" if pval < 0.05 else "Not significant — keep testing")
```

**Walkthrough.**
- `proportions_ztest` compares two conversion *rates* (proportions). For continuous metrics like revenue-per-user you'd use a t-test (`scipy.stats.ttest_ind`) instead.
- The `0.05` threshold (significance level) is convention, not law. It means "accept a 5% chance of a false positive." High-stakes decisions warrant a stricter threshold.
- **The discipline that actually matters:** decide your sample size *before* running (power analysis), don't peek and stop early when you see a result you like (that inflates false positives), and define your one primary metric up front. Most bad A/B tests fail on process, not math.

**Real-world case study — Bing's color-shade experiment.** One of the most-cited A/B testing stories at Microsoft: an engineer proposed slightly darkening the shade of blue used in Bing's search-result link titles — a change so trivial it nearly got deprioritized. Run as a controlled experiment on a slice of traffic, the tweak produced a measurable lift in user engagement and was estimated to be worth millions in additional annual revenue. The lesson companies took from it isn't "dark blue is good" — it's that human intuition is unreliable about what moves a metric, and that the only way to know is to randomize and measure. Big tech now runs thousands of concurrent experiments precisely because the wins are individually small, non-obvious, and undetectable without the statistical discipline A/B testing imposes.

**Where it breaks.** Needs enough traffic to detect realistic effects (small effects need large samples). Novelty effects can fool early reads. Running many tests at once inflates false positives unless you correct for it.

---

## 16. NLP Algorithms

**Business problem.** Make sense of unstructured text at scale: classify support tickets, gauge sentiment on reviews, extract entities from contracts, power chatbots. Supply chain example: parsing supplier emails or scanning contracts for risk clauses.

**Intuition.** Computers need text as numbers. The pipeline: clean text → tokenize (split into words/subwords) → vectorize (turn into numbers) → feed to a model. Classic vectorization is **TF-IDF**, which weights each word by how frequent it is in a document but down-weights words common across all documents (so "the" counts for little, "default" or "breach" count for a lot). The example does sentiment classification end to end.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

reviews = [
    "excellent product fast shipping highly recommend",
    "terrible quality broke immediately waste of money",
    "great value works perfectly very happy",
    "awful experience never buying again disappointed",
    "amazing fantastic love it best purchase",
    "poor build cheap materials returned it",
    "solid reliable does what it promises",
    "horrible defective unit total garbage",
]
sentiment = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]

# TF-IDF turns text into weighted word features; LogReg classifies sentiment
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), stop_words="english"),
    LogisticRegression(max_iter=1000)
)
model.fit(reviews, sentiment)

tests = ["wonderful quality very happy with purchase",
         "broke immediately complete waste"]
for t, p in zip(tests, model.predict(tests)):
    print(f"'{t}' -> {p}")
```

**Walkthrough.**
- `ngram_range=(1, 2)` captures single words *and* two-word phrases, so "not good" registers differently than "good" alone — important because negation flips meaning.
- `stop_words="english"` strips filler words (the, a, is) that carry no sentiment signal.
- TF-IDF + Logistic Regression is a strong, fast, interpretable baseline that still beats fancy models on many real classification tasks. Always build this before reaching for transformers.

**Real-world case study — routing inbound support at a SaaS company.** A software company receives 8,000 support emails a day across billing, technical bugs, account access, and feature requests. Manually triaging each to the right team takes minutes and delays the urgent ones. A TF-IDF + classifier pipeline reads the subject and body, predicts the category, and auto-routes — billing keywords ("invoice," "refund," "charge") pull toward the billing queue, error-related terms toward engineering. The model also scores sentiment so furious messages jump the queue regardless of category. The payoff is in time-to-first-response: tickets land on the right desk in seconds instead of waiting for a human router, and the team measures the win as a drop in median response time and a rise in first-contact resolution. A plain TF-IDF baseline often handles the bulk of this well enough that the fancier model isn't worth the added cost — which is exactly why you build the simple version first.

**Where it breaks.** Bag-of-words approaches ignore word order and context beyond n-grams; "the food was not bad" confuses them. They miss sarcasm and nuance. That gap is exactly what transformers closed — next.

---

## 17. Transformers

**Business problem.** State-of-the-art language understanding and generation: ChatGPT, translation, summarization, code copilots, semantic search. When context and meaning across a whole passage matter, not just keyword counts.

**Intuition.** The breakthrough is **attention** — when processing each word, the model looks at every other word and weighs how relevant each is to the current one. Processing "bank" in "river bank," attention lets the model weight "river" heavily and resolve the meaning. Unlike older sequential models that read word-by-word and forget, transformers see the whole sequence at once and learn rich, context-dependent representations. The encoder-decoder diagram in your slide: encoder digests the input, decoder generates the output.

```python
# pip install transformers torch
from transformers import pipeline

# Use small pretrained models so it runs without training anything yourself.
sentiment = pipeline("sentiment-analysis")
print(sentiment("The delivery was delayed but customer service was excellent")[0])

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
text = (
    "Global supply chains faced unprecedented disruption as port congestion, "
    "container shortages, and labor constraints compounded. Companies responded "
    "by diversifying suppliers, increasing safety stock, and investing in "
    "real-time visibility tools to anticipate and mitigate future shocks."
)
print(summarizer(text, max_length=40, min_length=15, do_sample=False)[0]["summary_text"])
```

**Walkthrough.**
- `pipeline(...)` downloads a pretrained transformer and wraps the whole tokenize → model → decode flow in one call. You're using models trained on enormous corpora — you benefit from that training without doing it yourself.
- This is **transfer learning**: the model already learned language broadly; you just apply it (or fine-tune on a small labeled set for your domain). It's why a few lines of code now do what took research teams years.
- The same architecture underlies LLMs. The difference is mostly scale — more parameters, more data, more compute — plus instruction tuning.

**Real-world case study — contract review in procurement.** A procurement team signs hundreds of supplier contracts a year, each 20–60 pages, and legal can't deeply review every one. Older keyword tools missed risk because the same clause gets worded a hundred ways — "either party may terminate" vs. "this agreement may be dissolved at the discretion of." A transformer-based model understands the *meaning*, not the literal words, so it can flag auto-renewal traps, unfavorable liability caps, and missing force-majeure clauses regardless of phrasing, then summarize each contract's key terms into a one-page brief. The reviewer reads the brief and the flagged sections instead of all 40 pages. The honest caveat that belongs in any deployment: the model can hallucinate a clause that isn't there or misread one that is, so its output is a *triage aid* that a human verifies on anything material — never the final word on a binding contract.

**Where it breaks.** Expensive to run and fine-tune. Can hallucinate (generate confident, wrong output). Inherits biases from training data. For a simple keyword-classification task, TF-IDF is cheaper and more controllable — match the tool to the problem.

---

## 18. Graph Algorithms

**Business problem.** When relationships *between* entities are the point, not the entities themselves: social networks, fraud rings (accounts connected through shared devices/addresses), supply chain dependency mapping, recommendation via connections. Tabular models miss structure that graphs make obvious.

**Intuition.** Model the world as **nodes** (entities) and **edges** (relationships). Then run algorithms over the structure: shortest path (cheapest route between two warehouses), centrality (which supplier is the most critical single point of failure), community detection (which accounts form a tight cluster — a possible fraud ring), PageRank (importance by connection). The supply chain example below finds the most critical node and the shortest path.

```python
# pip install networkx
import networkx as nx

# Supply network: edges weighted by lead time (days)
G = nx.DiGraph()
edges = [
    ("Supplier_A", "Factory", 5), ("Supplier_B", "Factory", 7),
    ("Factory", "Warehouse_1", 3), ("Factory", "Warehouse_2", 4),
    ("Warehouse_1", "Retailer_X", 2), ("Warehouse_2", "Retailer_X", 6),
    ("Warehouse_1", "Retailer_Y", 5),
]
G.add_weighted_edges_from(edges)

# Betweenness centrality: which node sits on the most critical paths?
central = nx.betweenness_centrality(G, weight="weight")
critical = max(central, key=central.get)
print("Most critical node (single point of failure):", critical)
print("Centrality scores:", {k: round(v, 3) for k, v in central.items()})

# Shortest (fastest) path from supplier to retailer
path = nx.shortest_path(G, "Supplier_A", "Retailer_X", weight="weight")
length = nx.shortest_path_length(G, "Supplier_A", "Retailer_X", weight="weight")
print(f"\nFastest route: {' -> '.join(path)}  ({length} days)")
```

**Walkthrough.**
- `betweenness_centrality` counts how often each node lies on shortest paths between other nodes. A high score means lots of flow depends on it — in supply chain, that's your single point of failure to build redundancy around. "Factory" scoring highest here is exactly the kind of risk insight tables won't surface.
- `shortest_path` with `weight` finds the lowest-total-lead-time route, not just the fewest hops — directly useful for routing and network design.
- **Graph Neural Networks** extend this to learning: they let you run ML *on* graph structure (node classification, link prediction) and power modern fraud detection and recommendations.

**Real-world case study — uncovering a fraud ring.** A bank's transaction monitoring kept clearing individual accounts because each, viewed alone, looked ordinary — small transfers, modest balances, nothing tripping a per-account threshold. Modeling customers, devices, phone numbers, and addresses as a graph changed the picture entirely: dozens of "unrelated" accounts turned out to share a handful of devices and mailing addresses, forming a tight, densely-connected community no single-account rule could ever see. Community-detection and centrality algorithms surfaced the ring as a structural pattern, and the shared-device nodes with high centrality were the hubs to freeze first. This is why graph analytics has become standard in financial-crime units — fraud is fundamentally relational, and the signal lives in the *connections* between entities, not in any one entity's row of data.

**Where it breaks.** Graph algorithms can be computationally heavy on massive networks (billions of edges need specialized graph databases like Neo4j). Building the graph correctly — deciding what's a node and what's an edge — is often the hardest part.

---

## 19. Survival Analysis

**Business problem.** "How long until X happens?" — and crucially, you're analyzing while some subjects haven't had the event yet. Time until customer churn, time until equipment failure, time until a patient relapses. The twist that breaks normal regression: **censoring** — many customers haven't churned *yet*, so you don't know their final lifetime, only that it's "at least this long."

**Intuition.** You can't just drop the not-yet-churned customers (that biases you toward short lifetimes) or pretend they churned now (false). Survival analysis handles censored data properly. The **Kaplan-Meier** curve (your slide's chart) estimates the probability of "surviving" past each time point. The **Cox model** goes further: it tells you which factors raise or lower the hazard (instantaneous risk) of the event.

```python
# pip install lifelines
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

rng = np.random.default_rng(2)
n = 300
tenure = rng.exponential(20, n).round(1)             # months observed
churned = rng.binomial(1, 0.6, n)                    # 1=churned, 0=still active (censored)
monthly_charge = rng.uniform(20, 120, n)

df = pd.DataFrame({"tenure": tenure, "churned": churned, "monthly_charge": monthly_charge})

# Kaplan-Meier: overall survival curve
km = KaplanMeierFitter()
km.fit(df["tenure"], event_observed=df["churned"])
print("Probability still a customer at:")
for t in [6, 12, 24]:
    print(f"  {t:2d} months: {km.predict(t):.1%}")

# Cox model: does monthly_charge affect churn risk?
cox = CoxPHFitter()
cox.fit(df[["tenure", "churned", "monthly_charge"]],
        duration_col="tenure", event_col="churned")
hr = np.exp(cox.params_["monthly_charge"])
print(f"\nHazard ratio per $1 monthly charge: {hr:.4f}")
print("(>1 means higher charge increases churn risk)")
```

**Walkthrough.**
- `event_observed` is the key column: 1 if the event happened, 0 if censored (still active at last observation). This is how the math correctly uses partial information instead of throwing it away.
- `km.predict(t)` reads the survival curve — "what fraction are still customers at month t." This directly feeds retention forecasting and customer lifetime value.
- The Cox **hazard ratio** is the actionable output: a ratio of 1.02 per dollar means each extra dollar of monthly charge raises churn risk by 2%. That's a pricing lever you can quantify.

**Real-world case study — wind-turbine gearbox maintenance.** An energy operator runs 400 wind turbines and wants to schedule gearbox maintenance before failure (a failed gearbox means a crane, days of downtime, and lost generation) but not so early that good components get replaced wastefully. The catch is censoring: most turbines in the fleet *haven't failed yet*, so a naive "average time to failure" computed only on the ones that broke is badly biased toward early failure. Survival analysis uses the still-running turbines correctly — they contribute the information "survived at least this long." The Kaplan-Meier curve gives the probability a gearbox is still healthy at each operating-hour milestone, and a Cox model quantifies how operating factors (average load, temperature cycling, site turbulence) raise or lower the failure hazard. Maintenance gets scheduled at the point where failure probability crosses the threshold that balances downtime cost against premature-replacement cost. The same machinery powers customer-churn timing and clinical relapse studies — anywhere the question is "how long until," with incomplete observations.

**Where it breaks.** The Cox model assumes hazards stay proportional over time (the "proportional hazards" assumption) — test it before trusting it. Heavy censoring or very few events makes estimates shaky.

---

## 20. Reinforcement Learning

**Business problem.** Sequential decision-making where each action changes the situation and you optimize a long-term reward, not a one-shot prediction: robotics, game AI, dynamic pricing, inventory replenishment policies, ad bidding. Supply chain fit: learning a reordering policy that balances stockouts against holding costs over time.

**Intuition.** An **agent** takes **actions** in an **environment**, receives **rewards**, and learns a **policy** (what to do in each state) that maximizes total reward over time. No labeled "correct answers" — the agent learns by trial and error, balancing **exploration** (try new things to discover what works) against **exploitation** (use what it already knows works). Q-learning, shown below, learns the value of each action in each state.

```python
import numpy as np

# Tiny inventory problem: states = stock level 0..5, actions = order 0..3 units
# Reward: meet demand (+), holding cost (-), stockout penalty (-)
rng = np.random.default_rng(0)
n_states, n_actions = 6, 4
Q = np.zeros((n_states, n_actions))
alpha, gamma, epsilon = 0.1, 0.9, 0.2   # learning rate, discount, exploration rate

def step(stock, order):
    demand = rng.integers(0, 4)
    new_stock = min(stock + order, n_states - 1)
    sold = min(new_stock, demand)
    reward = sold * 5 - new_stock * 1 - max(0, demand - new_stock) * 3
    return min(new_stock - sold, n_states - 1), reward

for episode in range(5000):
    stock = rng.integers(0, n_states)
    for _ in range(20):
        # epsilon-greedy: explore vs exploit
        action = rng.integers(n_actions) if rng.random() < epsilon else Q[stock].argmax()
        new_stock, reward = step(stock, action)
        # Q-learning update: blend old estimate with new experience
        Q[stock, action] += alpha * (reward + gamma * Q[new_stock].max() - Q[stock, action])
        stock = new_stock

print("Learned policy (best order quantity per stock level):")
for s in range(n_states):
    print(f"  Stock {s}: order {Q[s].argmax()} units")
```

**Walkthrough.**
- The **Q-table** stores the learned value of taking each action in each state. After training, the best policy is just "in each state, pick the highest-value action."
- The **epsilon-greedy** line is the exploration/exploitation balance in code: 20% of the time act randomly to discover, 80% of the time exploit the current best. Without exploration the agent gets stuck on the first decent strategy it finds.
- The **Q-learning update** is the heart: it nudges the current estimate toward `reward + discounted future value`. `gamma=0.9` means future rewards matter almost as much as immediate ones — the agent learns to plan ahead, not just grab the next reward.
- Notice the learned policy: order more when stock is low, less when high. The agent discovered a sensible inventory rule purely from rewards, never told the rule directly.

**Real-world case study — cooling Google's data centers.** Google's data centers consume enormous amounts of energy, much of it for cooling. DeepMind applied reinforcement learning to the cooling system: the agent observes sensor states (temperatures, pump speeds, weather) and adjusts cooling-equipment setpoints, with the reward tied to energy efficiency under hard safety constraints. Google reported the system cut the energy used for cooling by around 40% — a result no static rule-based controller had achieved, because the agent learned a policy that balances dozens of interacting variables in real time and adapts as conditions change. It captures every defining trait of RL: sequential decisions where each action changes the next state, a long-horizon reward (sustained efficiency, not one good reading), and a policy learned from experience rather than hand-coded. The same shape of problem — sequential decisions optimizing a long-run reward under constraints — is exactly what makes RL attractive for dynamic pricing and inventory-replenishment policy, where each order today changes tomorrow's stock position.

**Where it breaks.** Sample-hungry — needs huge numbers of trials, which is fine in simulation but expensive or dangerous in the real world. Reward design is treacherous: a poorly specified reward leads to the agent gaming it in unintended ways. Real problems use Deep RL (neural networks replacing the Q-table) for large state spaces.

---

## How to actually learn these

A sequence that builds properly, matched to your stated goal of moving toward applied AI/ML work:

1. **Get fluent with 1, 2, 3** (linear/logistic regression, decision trees). They teach supervised learning, the train/test discipline, and the metrics everything else reuses.
2. **Add 4, 10** (random forest, gradient boosting). These win on real tabular data — the bulk of business ML. XGBoost/LightGBM is the single most valuable practical skill here.
3. **Add 5, 8** (K-Means, PCA) for the unsupervised toolkit you'll use in segmentation and preprocessing.
4. **Pick depth based on your domain.** For supply chain specifically: 7 (forecasting), 18 (graphs), 19 (survival), and 20 (RL for inventory policy) map directly to real problems you'll recognize.
5. **Treat 12, 16, 17 as a track of their own** — neural nets → classic NLP → transformers. This is where the frontier and the job market are heaviest right now.

The thread through all of it: always start from the business problem, always hold out test data honestly, always check whether a simple model is good enough before reaching for a complex one. The complex model is rarely the bottleneck — the data quality and the problem framing usually are.
