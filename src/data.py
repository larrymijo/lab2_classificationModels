"""Loading and cleaning for the Lab 2 dataset (CarData / ALL_CARS_DATA.csv).

All cleaning decisions live here so the notebook stays readable and every run uses
exactly the same data. The reasoning behind each decision is in docs/step1-dataset-choice.md.
"""
import numpy as np

from src.overview import DATA_DIR, load_dataset

CSV_PATH = DATA_DIR / "CarData" / "ALL_CARS_DATA.csv"

RANDOM_STATE = 42
TARGET = "PRICE RANGE"
CLASS_ORDER = ["Low", "Medium", "High", "Luxury"]        # ordered from cheapest to most expensive

CATEGORICAL = ["MAKE ORIGIN", "BRAND", "CAR TYPE", "STOCK TYPE", "DEALER LOCATION (STATE)"]
SKEWED_NUMERIC = ["MILEAGE", "REVIEW"]                   # long right tails -> log transform
NUMERIC = ["AGE OF CAR", "RATING"]

LEAKY = ["PRICE($)"]                                     # PRICE RANGE is a binned version of the price
IDENTIFIERS = ["CAR ID", "IMAGE"]                        # unique per row, no predictive meaning
REDUNDANT = ["MODEL", "PARENT COMPANY"]                  # MODEL == 2024 - AGE OF CAR; PARENT COMPANY ~ BRAND
HIGH_CARDINALITY = ["MODEL/CLASS", "CAR NAME", "DEALER NAME", "DEALER LOCATION (CITY)"]

MISSING_CODE = -1                                        # used for "no rating" / "no reviews"
ORIGIN_FIXES = {"United States": "USA", "United Kingdom": "UK"}


def group_stock_type(series):
    """'New' and 'Used' stay; the 33 brand-specific '<Brand> Certified' values become 'Certified'."""
    return series.where(series.isin(["New", "Used"]), "Certified")


def load_cars(path=CSV_PATH, sample_size=40_000, random_state=RANDOM_STATE):
    """Load the CSV, apply the documented cleaning, and optionally take a random sample.

    Returns (df, notes): the cleaned DataFrame with the feature columns + TARGET,
    and a list of notes describing what each step changed.
    """
    df = load_dataset(path)
    notes = [f"Loaded {len(df):,} rows x {df.shape[1]} columns"]

    def step(mask, reason):
        nonlocal df
        removed = int((~mask).sum())
        df = df[mask]
        notes.append(f"{reason}: removed {removed:,} rows -> {len(df):,} left")

    step(df[TARGET].notna(), "Drop rows without a PRICE RANGE")
    # The same listing scraped twice gets a new CAR ID / IMAGE but is otherwise identical.
    # Keeping both copies could put one in training and its twin in the test set.
    content_columns = [c for c in df.columns if c not in IDENTIFIERS]
    step(~df.duplicated(subset=content_columns), "Drop duplicate listings (identical apart from CAR ID / IMAGE)")

    hidden_missing = {col: int((df[col] == MISSING_CODE).sum()) for col in ["RATING", "REVIEW"]}
    df = df.assign(**{col: df[col].replace(MISSING_CODE, np.nan) for col in ["RATING", "REVIEW"]})
    notes.append(f"Recoded -1 as missing: {hidden_missing}")

    df = df.assign(**{
        "MAKE ORIGIN": df["MAKE ORIGIN"].replace(ORIGIN_FIXES),
        "STOCK TYPE": group_stock_type(df["STOCK TYPE"]),
    })
    notes.append("Unified MAKE ORIGIN spellings (United States -> USA, United Kingdom -> UK)")
    notes.append("Grouped STOCK TYPE into New / Used / Certified")

    dropped = LEAKY + IDENTIFIERS + REDUNDANT + HIGH_CARDINALITY
    df = df.drop(columns=dropped)
    notes.append(f"Dropped columns: {dropped}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
        notes.append(f"Random sample of {sample_size:,} rows (random_state={random_state})")

    df = df[CATEGORICAL + SKEWED_NUMERIC + NUMERIC + [TARGET]].reset_index(drop=True)
    return df, notes


def load_raw_with_price(path=CSV_PATH):
    """Raw rows with PRICE($) kept - used only for the leakage demonstration."""
    df = load_dataset(path)
    return df[df[TARGET].notna()]
