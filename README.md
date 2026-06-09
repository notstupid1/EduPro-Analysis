# EduPro Analysis — Predictive Modeling for Online Course Performance

A machine learning pipeline that predicts **enrollment counts** and **total revenue** for courses on the EduPro online learning platform. The project covers the full data science workflow: raw data ingestion, feature engineering, model training, evaluation, and an interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Features](#features)
- [Models](#models)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data](#data)

---

## Overview

EduPro Analysis processes multi-sheet Excel data (courses, teachers, transactions, users) and builds regression models to forecast two key business metrics:

- **Enrollment Count** — how many students are likely to enroll in a course
- **Total Revenue** — projected revenue a course will generate

Results are surfaced through a Streamlit web app for interactive exploration.

---

## Project Structure

```
EduPro-Analysis/
│
├── data/
│   ├── raw/
│   │   └── EduPro Online Platform.xlsx   # Source data (4 sheets)
│   └── processed/
│       ├── merged_data.csv               # Output of data_preparation.py
│       └── features_engineered.csv       # Output of feature_engineering.py
│
├── info
│   ├── executive_summary.md              # Summary for executives
│   └── research_paper.md                 # Research Paper
│
├── models/
│   └── *.pkl                             # Saved trained model files
│
├── data_preparation.py                   # Data loading, merging, aggregation
├── feature_engineering.py                # Feature creation and encoding
├── modeling.py                           # Model training and evaluation
├── README.md                             # Readme file
└── streamlit_app.py                      # Interactive dashboard
```

---

## Pipeline

The project runs in three sequential stages:

```
data_preparation.py  →  feature_engineering.py  →  modeling.py  →  streamlit_app.py
```

**1. Data Preparation** — Loads the four Excel sheets, aggregates transaction-level data to the course level (total revenue, enrollment count, course age), resolves teacher assignments, and saves a merged CSV.

**2. Feature Engineering** — Builds structured features on top of the merged data: price bands, duration buckets, rating tiers, experience levels, category aggregates, teacher workload, and one-hot encodings. Normalizes numerical columns and drops highly correlated features (threshold: 0.95).

**3. Modeling** — Trains five regression models for each of the two targets (enrollment and revenue), evaluates them with MAE, RMSE, and R², runs 5-fold cross-validation, and saves all models as `.pkl` files.

**4. Streamlit App** — Provides an interactive UI for exploring predictions and results.

---

## Features

The feature engineering stage creates the following categories of features:

| Category | Features |
|---|---|
| Price | `price_band` (low / medium / high) |
| Duration | `duration_bucket` (short / medium / long) |
| Ratings | `course_rating_tier`, `teacher_rating_tier` (poor / average / good / excellent) |
| Experience | `experience_level` (junior / mid / senior) |
| Encoding | `course_level_encoded`, `course_category_encoded` |
| Aggregates | `avg_category_rating`, `avg_category_enrollment`, `avg_category_revenue` |
| Teacher | `teacher_courses_count`, `expertise_category_match` |
| Time | `first_transaction_month` |
| Normalized | `*_normalized` versions of all numerical columns |

---

## Models

Five regression algorithms are trained for each target variable (10 models total):

| Model | Enrollment | Revenue |
|---|---|---|
| Linear Regression | ✓ | ✓ |
| Ridge Regression (α=1.0) | ✓ | ✓ |
| Lasso Regression (α=0.1) | ✓ | ✓ |
| Random Forest (100 trees, max depth 15) | ✓ | ✓ |
| Gradient Boosting (100 trees, lr=0.1) | ✓ | ✓ |

Evaluation metrics reported: Train MAE, Test MAE, Test RMSE, Train R², Test R², and 5-fold CV R².

Feature importance is extracted from the tree-based models (Random Forest and Gradient Boosting) for interpretability.

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn openpyxl joblib streamlit
```

### Installation

```bash
git clone https://github.com/notstupid1/EduPro-Analysis.git
cd EduPro-Analysis
```

Place the raw Excel file at:
```
data/raw/EduPro Online Platform.xlsx
```

---

## Usage

Run each stage in order:

```bash
# Step 1: Prepare and merge data
python data_preparation.py

# Step 2: Engineer features
python feature_engineering.py

# Step 3: Train and evaluate models
python modeling.py

# Step 4: Launch the dashboard
streamlit run streamlit_app.py
```

Trained models are saved to `./models/` as `.pkl` files and can be loaded with `joblib.load()` for inference.

---

## Data

The raw data is an Excel workbook with four sheets:

| Sheet | Description |
|---|---|
| `Courses` | Course metadata (ID, name, category, level, price, duration, rating) |
| `Teachers` | Teacher profiles (ID, name, expertise, experience, rating) |
| `Transactions` | Enrollment transactions (course, teacher, user, date, amount) |
| `Users` | User/student information |

The pipeline resolves teacher-course assignments by selecting the teacher most frequently associated with each course in the transaction history.
