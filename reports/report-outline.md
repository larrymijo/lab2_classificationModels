# Lab 2 report — outline with results

DA380A Machine Learning · Group 28 · Lab 2: Classification Models

The PDF should **interpret and compare** the models, not repeat the code. Every number below comes from `notebooks/lab2_classification.ipynb` (the section is given in brackets). Aim for 3–5 pages with 2–4 figures. Write it in your own words: you must be able to explain every sentence.

## 1. Introduction (½ page)
- Dataset: `CarData` — 266,847 car listings from cars.com (brand, origin, type, age, mileage, stock type, rating, reviews, dealer state, price).
- Target: `PRICE RANGE`, 4 classes (Low, Medium, High, Luxury) = how expensive a car is **compared with other cars of the same type**. Predicting it estimates a car's price tier without knowing the price.
- Why this dataset: balanced 4-class target, clean features, understandable domain (details in `docs/step1-dataset-choice.md`).

## 2. Data and preparation (1 page)
- Removed 25,919 duplicate listings (same car scraped twice) and 1 unlabelled row → 240,927 listings; random sample of **40,000**. [1.2, 1.4]
- Hidden missing values: `RATING = −1` for ≈ 20% of rows → recoded as missing, imputed with the median plus a "was missing" flag. [1.3, 2]
- Unified country spellings; grouped 35 stock types into New / Used / Certified. [1.3]
- **Leakage:** `PRICE RANGE` is made from the price (price quartile within the car type reproduces 82.1% of labels) → `PRICE($)` removed. With only `CAR TYPE` + `PRICE($)` a forest gets **88%** accuracy. [1.3, 7.4]
- Classes balanced (24–26%) → accuracy is meaningful; **macro** averages used for precision/recall/F1 so every class counts equally. [1.5, 4]
- Exploration findings: stock type is the strongest single signal (used cars 45% *Low*, new cars 44% *Luxury*); *High* and *Luxury* cars are both mostly new with 0 miles. [1.6–1.8]
- Stratified 80/20 split (32,000 / 8,000); 5-fold cross-validation on the training set. [3]
- Suggested figure: class mix per stock type [1.7].

## 3. Results overview

Test set (8,000 cars), macro averages [5, 6]:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy (most frequent) | 0.264 | 0.066 | 0.250 | 0.104 |
| Logistic Regression | 0.577 | 0.561 | 0.572 | 0.562 |
| Logistic Regression (tuned, C = 10) | 0.576 | 0.562 | 0.571 | 0.563 |
| Random Forest | 0.637 | 0.632 | 0.635 | 0.633 |
| Random Forest (tuned) | **0.642** | **0.638** | **0.639** | **0.639** |

Tuning: logistic regression unchanged (it doesn't overfit, so less regularisation can't help); forest +0.5 points with `max_features = 0.3`, `min_samples_leaf = 3`, 300 trees. [6]

## 4. Question 1 — which classes are predicted well and which are difficult?

F1 per class, tuned models [7.1]:

| Class | Logistic Regression | Random Forest |
|---|---|---|
| Low | 0.783 | 0.805 |
| Medium | 0.490 | 0.559 |
| High | **0.361** | **0.516** |
| Luxury | 0.616 | 0.675 |

- **Low is easy** (≈ 81% found by both): cheap cars are older, used, high-mileage.
- **High is hardest**: logistic regression labels 43% of *High* cars as *Luxury*; the forest finds 51% of them. *High* and *Luxury* cars are both mostly new with zero mileage, so only brand / type / origin can separate them.
- **Mistakes are mostly near misses**: 81% (forest) and 78% (logistic regression) of errors are one class off. [7.1]
- Suggested figure: the two confusion matrices [5 or 7.1].

## 5. Question 2 — strengths and weaknesses
Use the table in notebook 7.5:
- **Logistic Regression:** fast, interpretable, gives probabilities, no overfitting (train 0.590 vs test 0.576) — but linear boundaries can't capture interactions, so it collapses *High* into *Luxury*.
- **Random Forest:** better on every class, captures interactions — `CAR TYPE` is useless alone but the forest's 2nd most important feature, because the class is defined relative to the type; but slower, less transparent and overfits (train 0.821 vs test 0.642).
- Feature importance [7.2]: `MILEAGE` first for both (accuracy drop 0.30 LR / 0.16 RF); then `CAR TYPE` (0.13), `AGE OF CAR` (0.10), `BRAND` (0.08) for the forest; `RATING`/`REVIEW` ≈ 0.
- Suggested figure: permutation importance [7.2].

## 6. Question 3 — recommendation
- Recommend the **tuned Random Forest**: accuracy 0.642 / macro F1 0.639 vs 0.576 / 0.563; 2,865 vs 3,396 errors out of 8,000; the biggest gains are on the hard middle classes. [7.6]
- Limitations: 40,000-row sample; model name and dealer city left out; the label is relative to the car type.
- Improvements: all rows, model name with rare models grouped, gradient boosting, an ordinal approach since the classes are ordered.

## 7. Conclusion (3–4 sentences)
What you did, the main result, which model won and why, one improvement.
