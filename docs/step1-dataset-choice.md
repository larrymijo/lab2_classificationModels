# Step 1 — Choosing the dataset

**Decision: `CarData` (`ALL_CARS_DATA.csv`), target `PRICE RANGE` — 4 classes: Low, Medium, High, Luxury.**

## The four candidates

| Dataset | File | Rows × columns | Possible target | Verdict |
|---|---|---|---|---|
| perfumes | `fra_perfumes.csv` | 70,103 × 8 | `Gender` (3 classes) | The useful feature, `Main Accords`, is a list stored as text (`"['citrus', 'musky', …]"`) that must be parsed into features first; 9% missing ratings; 112 duplicates |
| NetworkEvent | `HealthCareData2024.csv` | 568,309 × 14 | `Classification` (Normal / Malicious) | **96% / 4% imbalance**; `SystemAccessRate` 60% missing; inconsistent labels (`PolicyViolation` vs `Policy_Violation`, `Info` vs `Informational`); `-1` values in `NetworkAccessFrequency` |
| musicdata | `comercial.csv` | 478,103 × 19 | none obvious (`mode` = major/minor?) | Radio-play log of ~600 songs repeated over time — no natural class to predict |
| **CarData** | **`ALL_CARS_DATA.csv`** | **266,847 × 19** | **`PRICE RANGE` (4 classes)** | **Chosen** |

## Scoring (1 = poor, 3 = good)

| Criterion | perfumes | NetworkEvent | musicdata | CarData |
|---|---|---|---|---|
| Clear class target | 3 | 3 | 1 | 3 |
| Enough rows | 3 | 3 | 3 | 3 |
| Useful features without heavy parsing | 1 | 3 | 2 | 3 |
| Data quality | 2 | 1 | 3 | 2 |
| Class balance | 2 | 1 | 2 | 3 |
| We understand the domain | 2 | 1 | 2 | 3 |
| **Total** | **13** | **12** | **13** | **17** |

CarData wins clearly: a ready-made, **balanced 4-class target** (Low 23.7% · Medium 24.7% · High 25.0% · Luxury 26.6% in the raw data), a good mix of numeric and categorical features, almost no missing values, and a domain everyone understands.

## What the dataset contains

Car listings scraped from cars.com (2024).

| Column | Type | Notes |
|---|---|---|
| `CAR ID`, `IMAGE` | identifiers | unique per listing → dropped |
| `PARENT COMPANY` | categorical (52) | almost the same information as `BRAND` → dropped |
| `MAKE ORIGIN` | categorical | `USA`/`United States` and `UK`/`United Kingdom` spelled two ways → unified (10 origins) |
| `BRAND` | categorical (88) | kept |
| `MODEL/CLASS`, `CAR NAME` | text (1,225 / 8,268 values) | too many categories → dropped |
| `MODEL` | model year (1919–2025) | `AGE OF CAR == 2024 − MODEL` for every row → dropped (kept `AGE OF CAR`) |
| `CAR TYPE` | categorical (29) | kept |
| `PRICE RANGE` | **target** | Low / Medium / High / Luxury |
| `PRICE($)` | numeric | **leakage** → dropped (see below) |
| `AGE OF CAR` | numeric | −1 for 2025 models (real, not an error) |
| `STOCK TYPE` | categorical (35) | `New`, `Used` + 33 `<Brand> Certified` values → grouped into New / Used / Certified |
| `MILEAGE` | numeric | 0 for every new car; up to ≈ 900,000 |
| `RATING` | numeric | **−1 = "no rating"** (50,289 rows in the raw data) → recoded as missing |
| `REVIEW` | numeric | **−1 = "no reviews"** (43 rows) → recoded as missing |
| `DEALER NAME`, `DEALER LOCATION (CITY)` | text (14,281 / 3,901 values) | too many categories → dropped |
| `DEALER LOCATION (STATE)` | categorical (52) | kept |

## The target

`PRICE RANGE` puts each listing in one of four price tiers. The price ranges **overlap** between classes (a *Low* car can cost up to $61k, a *Luxury* car as little as $33k), so it isn't a fixed cut-off. Inside each big car type the four classes are ≈ 25% each, and the rule *"rank the price among cars of the same type and cut into quarters"* reproduces the label for **82.1%** of rows. So `PRICE RANGE` means: **how expensive the car is compared with other cars of the same type.** Predicting it estimates a car's price tier from its characteristics, without knowing the price.

## Cleaning decisions (implemented in `src/data.py`)

| Decision | Rows affected | Why |
|---|---|---|
| Drop the row without a `PRICE RANGE` | −1 | No target |
| Drop listings identical apart from `CAR ID` / `IMAGE` | −25,919 | The same car scraped twice; both copies could end up in training and test |
| Recode `RATING` / `REVIEW` = −1 as missing | 48,181 / 43 (after dedup) | −1 is a code for "none yet", not a real value; imputed with the median + a "was missing" indicator in the pipeline |
| Unify `MAKE ORIGIN` spellings | — | Same country under two names would become two different categories |
| Group `STOCK TYPE` into New / Used / Certified | — | 33 brand-specific certified labels mean the same thing |
| Drop `PRICE($)` | — | **Leakage** — the target is made from it. With only `CAR TYPE` + `PRICE($)` a forest reaches 88% accuracy (notebook 7.4) |
| Drop IDs, `MODEL`, `PARENT COMPANY`, model/dealer names and city | — | Identifiers, redundant, or thousands of categories |
| Random sample of 40,000 rows (`random_state=42`) | 240,927 → 40,000 | Keeps cross-validation and grid search to a few minutes. Documented in the report |

**Final features:** `MAKE ORIGIN`, `BRAND`, `CAR TYPE`, `STOCK TYPE`, `DEALER LOCATION (STATE)` (one-hot encoded) and `MILEAGE`, `REVIEW` (log + standardised), `AGE OF CAR`, `RATING` (standardised; missing values imputed).
