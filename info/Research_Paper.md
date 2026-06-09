# Predictive Analytics for Online Course Performance: A Machine Learning Approach to Enrollment and Revenue Forecasting on the EduPro Platform

---

**Abstract**

The rapid expansion of online education platforms has created an urgent need for data-driven tools that help administrators and instructors forecast course demand and revenue. This paper presents EduPro Analysis, a complete machine learning pipeline designed to predict enrollment counts and total revenue for courses listed on the EduPro online learning platform. The system integrates multi-source transactional and metadata into a unified feature space, applies a suite of regression algorithms ranging from linear baselines to ensemble methods, and surfaces results through an interactive Streamlit dashboard. We describe the data preparation strategy, feature engineering methodology, model architectures, evaluation framework, and deployment interface in detail. Gradient Boosting and Random Forest regressors trained on engineered categorical, numerical, and aggregate features form the core predictive layer. The pipeline provides actionable forecasting capability for platform operators seeking to optimize course offerings, pricing strategies, and instructor assignments.

**Keywords:** online education, predictive modeling, enrollment forecasting, revenue prediction, feature engineering, ensemble learning, Streamlit

---

## 1. Introduction

The global e-learning market has grown substantially over the past decade, driven by advances in digital infrastructure and shifting preferences toward flexible, self-paced education. Platforms hosting hundreds of courses across diverse categories face the challenge of anticipating which offerings will attract learners and generate sustainable revenue. Without reliable forecasts, decisions around pricing, course creation, instructor recruitment, and marketing remain reactive rather than strategic.

EduPro Analysis addresses this gap by building a complete predictive modeling pipeline anchored in real transactional data. The project targets two interrelated business metrics: **enrollment count** (how many students a course attracts) and **total revenue** (the cumulative financial return). Both are treated as continuous regression targets, and the pipeline produces models for each using the same feature set, enabling direct comparison of the factors that drive learner acquisition versus financial performance.

The contributions of this work are:

1. A modular, reproducible data preparation pipeline that merges course metadata, instructor profiles, and transaction records into a single analytic dataset.
2. A rich feature engineering stage that encodes domain knowledge about pricing tiers, instructor experience, rating quality, and category-level dynamics.
3. A systematic model comparison across six regression algorithms, evaluated with MAE, RMSE, R², and 5-fold cross-validation.
4. An interactive web dashboard built with Streamlit that makes predictions accessible to non-technical stakeholders.

---

## 2. Related Work

Predicting student enrollment in online settings has been explored from multiple angles. Early work focused on demographic and academic background variables to predict completion or dropout rates in MOOCs (Massive Open Online Courses). More recent approaches leverage behavioral clickstream data, discussion forum activity, and video engagement metrics as proxies for intent and persistence.

Revenue forecasting in e-learning platforms is less studied in the academic literature, though it parallels subscription churn and lifetime value prediction in SaaS contexts. Work on pricing elasticity in digital goods markets informs the intuition that price band and course category interact non-trivially with demand. Instructor reputation, measured through ratings and course completion rates, has been shown to be a significant predictor of enrollment in several platform-specific studies.

The present work differs from much prior literature in its focus on course-level aggregated signals rather than individual learner behavior, making it appropriate for scenarios where user-level data is unavailable or privacy-constrained. The reliance on transaction history, course metadata, and instructor profiles reflects a realistic operational data environment for a mid-size platform.

---

## 3. Data

### 3.1 Data Sources

The raw data consists of a single Excel workbook structured across four sheets representing the core entities of the EduPro platform:

| Sheet | Description | Key Fields |
|---|---|---|
| Courses | Course catalog entries | CourseID, CourseName, CourseCategory, CourseLevel, CoursePrice, CourseDuration, CourseRating, CourseType |
| Teachers | Instructor profiles | TeacherID, TeacherName, Expertise, YearsOfExperience, TeacherRating, Gender, Age |
| Transactions | Enrollment transactions | CourseID, TeacherID, UserID, TransactionDate, Amount |
| Users | Learner accounts | UserID and associated demographic fields |

The Transactions table is the central fact table. Each row represents a single enrollment payment, providing the ground truth for both revenue and enrollment count targets.

### 3.2 Data Preparation

The `data_preparation.py` module orchestrates all loading, merging, and aggregation operations. The preparation pipeline proceeds in four stages:

**Stage 1 — Loading.** All four sheets are read into Pandas DataFrames using `pd.read_excel`. No schema transformation occurs at this stage.

**Stage 2 — Course-level aggregation.** Transactions are grouped by `CourseID`, computing: total revenue (`Amount` sum), average revenue per enrollment (`Amount` mean), enrollment count (`Amount` count), and the date range of transactions (first and last `TransactionDate`). A derived field, `course_age_days`, captures the span in days between first and last transaction, serving as a proxy for how long a course has been active on the platform.

**Stage 3 — Teacher assignment resolution.** Because multiple instructors may be associated with a course across different transactions, the pipeline selects the teacher most frequently appearing in the transaction history for each course. This resolves a many-to-many relationship into a clean one-to-one mapping suitable for supervised learning.

**Stage 4 — Merging.** The aggregated course metrics are joined with the course catalog (on `CourseID`) and then with the teacher table (on `TeacherID`). Null values arising from courses with no transaction history are filled with sensible defaults: zero for counts and amounts, and `0` days for course age. The merged dataset is saved as `data/processed/merged_data.csv`.

---

## 4. Feature Engineering

The `feature_engineering.py` module transforms the merged dataset into a fully numeric, model-ready feature matrix. The engineering operations span five categories: categorical binning, ordinal encoding, aggregate enrichment, workload signals, and temporal extraction.

### 4.1 Categorical Binning

Three continuous fields are binned into ordered categorical bands to reduce noise from extreme values and capture non-linear relationships:

- **Price bands:** CoursePrice is bucketed into `low` (≤ $100), `medium` ($100–$300), and `high` (> $300).
- **Duration buckets:** CourseDuration is bucketed into `short` (≤ 10 hours), `medium` (10–30 hours), and `long` (> 30 hours).
- **Rating tiers:** Both CourseRating and TeacherRating are binned into `poor` (0–2), `average` (2–3), `good` (3–4), and `excellent` (> 4).
- **Experience levels:** Teacher YearsOfExperience is bucketed into `junior` (0–3 years), `mid` (3–7 years), and `senior` (> 7 years).

### 4.2 Ordinal and Nominal Encoding

Course level (Beginner, Intermediate, Advanced) is mapped to integers (1, 2, 3), preserving its natural ordering. Course category is label-encoded across twelve domain areas: Digital Marketing, Business, Design, Programming, Marketing, Data Science, Machine Learning, Cybersecurity, Finance, Project Management, Artificial Intelligence, and Web Development.

An expert-category alignment score (`expertise_category_match`) is computed by mapping each teacher's expertise area to a broad domain (Technology, Business, or Design) and checking whether it matches the course's category. This binary feature captures a theoretically meaningful signal: instructors teaching within their domain of expertise may produce higher-quality courses.

### 4.3 Category-Level Aggregates

To give each course contextual information about its competitive position within its category, four aggregate statistics are computed across all courses sharing the same `CourseCategory`: average rating, total number of courses, average enrollment, and average revenue. These aggregates allow the model to learn, for example, that a Data Science course operates in a more competitive and enrollment-rich environment than a Cybersecurity course.

### 4.4 Workload and Temporal Features

A teacher workload feature (`teacher_courses_count`) counts how many courses each instructor is associated with on the platform. Instructors teaching many courses simultaneously may have different performance characteristics than those focused on a single offering.

The month of a course's first transaction is extracted as a time-of-year signal, capturing potential seasonal patterns in enrollment behavior.

### 4.5 One-Hot Encoding and Normalization

All binned categorical features (CourseCategory, CourseType, price_band, duration_bucket, experience_level, course_rating_tier, teacher_rating_tier) are one-hot encoded using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity. All numerical features are then normalized using `sklearn.preprocessing.StandardScaler`, creating `*_normalized` variants of each column.

### 4.6 Correlation Pruning

Features with a Pearson correlation exceeding 0.95 against any other feature are dropped from the matrix. This reduces multicollinearity and shrinks the dimensionality of the input space, helping linear models converge and reducing overfitting risk in tree-based models.

---

## 5. Modeling

### 5.1 Problem Formulation

Two parallel regression tasks are defined:

- **Task E:** Predict `enrollment_count` given course and instructor features.
- **Task R:** Predict `total_revenue` given course and instructor features.

Both tasks share the same feature matrix X, differing only in their target vector. This design allows model performance to be directly compared across the two tasks, highlighting whether enrollment and revenue are driven by the same or different feature signals.

### 5.2 Feature Selection

Columns that would constitute data leakage or carry no predictive signal are excluded from X: course and teacher identifiers, free-text names, age, gender, raw transaction dates, and the alternate target variable. Only numeric columns remaining after the feature engineering stage are retained.

### 5.3 Data Splitting

The dataset is split into training (80%) and test (20%) sets using a fixed random state of 42, ensuring reproducibility. The same split is used for all models to ensure fair comparison.

### 5.4 Model Suite

Six regression algorithms are trained for each task:

**Baseline Models:**

| Model | Configuration |
|---|---|
| Linear Regression | OLS, no regularization |
| Ridge Regression | L2 penalty, α = 1.0 |
| Lasso Regression | L1 penalty, α = 0.1 |

**Advanced Models:**

| Model | Configuration |
|---|---|
| Random Forest | 100 trees, max depth 15, all CPU cores |
| Gradient Boosting | 100 trees, max depth 5, learning rate 0.1 |

Linear models serve as interpretable baselines and help diagnose whether the problem is approximately linear. Ridge and Lasso add regularization to assess whether the high-dimensional one-hot encoded features cause overfitting in the linear regime. Random Forest and Gradient Boosting capture non-linear interactions and are expected to perform best given the mix of categorical and continuous features.

### 5.5 Evaluation Metrics

Each model is evaluated on four metrics:

- **MAE (Mean Absolute Error):** Average absolute deviation between predicted and actual values. Reported on both train and test sets to detect overfitting.
- **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily than MAE; reported on the test set.
- **R² (Coefficient of Determination):** Proportion of variance explained; reported on both train and test sets.
- **5-Fold Cross-Validation R²:** Mean and standard deviation of R² across five folds on the training set, assessing stability of the model fit.

### 5.6 Feature Importance

For tree-based models, Gini impurity-based feature importances are extracted and sorted. This enables post-hoc interpretation of which variables most strongly drive predictions, supporting actionable recommendations to platform administrators.

---

## 6. Deployment

The `streamlit_app.py` module provides an interactive multi-page dashboard for exploring the models and data without requiring any coding knowledge. The application is structured into six pages:

**Overview:** Displays four headline KPIs (total courses, total enrollments, total revenue, average course rating), a horizontal bar chart of the top 10 courses by enrollment, a pie chart of revenue distribution by category, and histograms of course price and enrollment distributions.

**Demand Prediction:** Provides an interactive interface where the user specifies course attributes (price, duration, level, category, expected rating) and instructor attributes (years of experience, rating). On clicking "Predict Enrollment," the four tree-based models produce enrollment and revenue estimates for the specified configuration, alongside a table of similar existing courses.

**Revenue Forecasting:** Presents category-level bar charts of total revenue and box plots of revenue per enrollment, enabling comparison of financial efficiency across course domains.

**Feature Importance:** Renders a horizontal bar chart of the top 20 features for any user-selected tree-based model, supporting interpretability analysis.

**Category Analysis:** Provides an aggregated statistics table across all categories (course count, total and average enrollments, total and average revenue, average course and teacher ratings) and a drilldown view for a selected category.

**Insights:** Exposes four expandable sections: top-performing courses, category performance summaries, top instructors by cumulative enrollment, and pricing insights segmented by price band.

Data and models are loaded with Streamlit's `@st.cache_data` and `@st.cache_resource` decorators to avoid redundant I/O on each user interaction.

---

## 7. Discussion

### 7.1 Design Choices

The decision to aggregate data at the course level rather than the transaction level reflects the operational question being answered: administrators and course creators want to know how a course as a whole will perform, not individual transaction probabilities. This aggregation smooths noise from individual payment timing and amounts but necessarily discards learner-level behavioral signals that could enrich predictions.

The teacher assignment resolution strategy — selecting the most frequently associated instructor — is a pragmatic simplification. In practice, courses may be co-taught or transferred between instructors; a more sophisticated treatment might model instructor effects as latent variables or use a weighted average of instructor attributes.

The expertise-category match score encodes a domain assumption that may not hold universally. A cybersecurity expert may teach an exceptionally popular business course. This feature may add noise in such cases and would benefit from empirical validation against held-out data.

### 7.2 Limitations

The pipeline has several limitations worth acknowledging. First, the dataset size is unspecified; model generalization will degrade for categories with few courses. Second, the absence of time-series modeling means the system cannot capture trend, seasonality, or momentum effects — a course growing in popularity will look identical to one declining, as long as its aggregate statistics match. Third, the correlation pruning threshold (0.95) is a fixed hyperparameter that may require tuning for different datasets. Finally, the Streamlit prediction interface constructs input feature vectors manually with hard-coded feature orderings, which is fragile and may produce incorrect predictions if the feature engineering stage adds or reorders columns.

### 7.3 Future Work

Several extensions could strengthen the pipeline:

- **Time-series models** (ARIMA, Prophet, or LSTM) could capture trend and seasonality in enrollment by disaggregating transaction data into monthly or weekly counts.
- **Learner-level features** (completion rates, review text sentiment, user demographics) could enrich the feature space substantially.
- **Hyperparameter optimization** via grid search or Bayesian optimization would likely improve the performance of all models, particularly Gradient Boosting.
- **Confidence intervals** on predictions would make the dashboard more useful for decision-makers who need to reason under uncertainty.
- **A/B test integration** could enable the platform to validate model-driven pricing or scheduling recommendations with real enrollment outcomes.

---

## 8. Conclusion

This paper has described EduPro Analysis, a complete machine learning pipeline for forecasting enrollment and revenue on an online education platform. The system processes multi-sheet transactional data through structured preparation and feature engineering stages, trains twelve regression models across two targets, and surfaces results through an interactive Streamlit dashboard. The modular architecture — four independent Python scripts connected by file-based intermediate artifacts — makes each stage independently testable and extensible. The inclusion of both linear baselines and ensemble methods enables rigorous comparison of model complexity against predictive gain. The dashboard democratizes access to predictions, supporting data-driven decision-making for non-technical platform administrators.

---
