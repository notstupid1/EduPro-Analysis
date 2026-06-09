# EduPro Analysis — Executive Summary

**Prepared for:** Platform Stakeholders & Leadership
**Project:** EduPro Demand & Revenue Forecasting
**Repository:** https://github.com/notstupid1/EduPro-Analysis

---

## What This Project Does

EduPro Analysis is a predictive intelligence system built for the EduPro online learning platform. It answers two critical business questions before a course goes to market or a decision is made:

> **How many students will enroll in a course — and how much revenue will it generate?**

By training machine learning models on historical enrollment and payment data, the system provides data-backed forecasts that replace guesswork in course planning, instructor assignment, and pricing decisions. Results are accessible through a live interactive dashboard — no technical background required.

---

## The Business Problem

Without forecasting tools, platform operators face three recurring challenges:

**Demand uncertainty** — New courses are launched without visibility into whether they will attract five students or five hundred, making resource allocation difficult.

**Pricing misalignment** — Courses priced too high depress enrollment; priced too low, they underperform financially. The optimal price point varies by category, instructor quality, and course duration.

EduPro Analysis addresses them by surfacing the historical patterns that predict success.

---

## What Data Powers the System

The models draw from four internal data sources that the platform already collects:

| Source | What It Contains |
|---|---|
| Course catalog | Name, category, level, price, duration, and rating for every course |
| Instructor profiles | Teaching experience, expertise area, and instructor rating |
| Transaction records | Every enrollment payment — date, amount, course, and instructor |
| User accounts | Learner registration data |

No external data is required. The system is entirely self-contained within EduPro's existing operational data.

---

## How the Forecasting Works

The pipeline runs in four stages, each building on the last:

**1. Data Integration** — Transaction records are aggregated per course to compute total revenue, total enrollments, average revenue per learner, and course age. Each course is then matched with its primary instructor.

**2. Feature Engineering** — Over twenty predictive signals are constructed from raw data. These include price tiers (low / medium / high), instructor experience levels (junior / mid / senior), course and instructor rating quality bands, category-level benchmarks (how does this course compare to others in its domain?), and instructor workload.

**3. Model Training** — Five regression algorithms are trained for each of the two targets — enrollment count and total revenue — producing ten models in total. These range from interpretable linear models to high-accuracy ensemble methods (Random Forest and Gradient Boosting).

**4. Interactive Dashboard** — All models and data are surfaced through a Streamlit web application that any team member can use without writing a single line of code.

---

## Key Capabilities of the Dashboard

| Page | What Stakeholders Can Do |
|---|---|
| Overview | See total platform enrollments, revenue, course count, and average ratings at a glance |
| Demand Prediction | Simulate a new course by entering its price, duration, level, category, and instructor profile — and instantly see predicted enrollment and revenue |
| Revenue Forecasting | Compare revenue performance and revenue-per-enrollment across all course categories |
| Feature Importance | Understand which factors most strongly drive enrollment and revenue predictions |
| Category Analysis | Drill into any course category to see all courses, their performance, and category-level benchmarks |
| Insights | Review top-performing courses, highest-impact instructors, and pricing patterns |

---

## What the Models Can Predict

When a course planner enters a course configuration into the dashboard, the system returns:

- **Predicted enrollment count** — estimated number of students likely to enroll
- **Predicted total revenue** — estimated cumulative revenue from the course
- **Comparable courses** — existing courses with similar attributes, for benchmarking

Four model variants (Random Forest and Gradient Boosting for each target) produce independent estimates, giving decision-makers a range rather than a single point forecast.

---

## Factors That Most Influence Performance

Based on the feature engineering and model architecture, the signals with the strongest expected influence on course outcomes are:

**Enrollment drivers**
- Course rating and instructor rating — quality signals that learners trust when choosing
- Category-level average enrollment — courses in high-demand categories inherit a structural advantage
- Price band — lower-priced courses generally attract broader enrollment
- Course duration — length affects perceived commitment and accessibility

**Revenue drivers**
- Course price — the dominant direct lever on revenue
- Enrollment volume — revenue scales with learner count
- Category average revenue — some domains command higher willingness-to-pay
- Instructor experience — senior instructors correlate with higher-value course positioning

---

## Strategic Implications

The system enables concrete operational improvements:

**Smarter course launches** — Before publishing a new course, planners can test different pricing and duration combinations in the dashboard and select the configuration that optimizes for enrollment, revenue, or both.

**Category portfolio planning** — The category analysis and revenue forecasting pages reveal which domains are over- or under-served relative to demand. Leadership can use this to prioritize content acquisition in high-potential, underpopulated categories.

---

## Technical Foundation

The system is built entirely in Python using industry-standard open-source libraries — pandas and scikit-learn for data processing and modeling, Streamlit for the dashboard, and joblib for model persistence. All trained models are saved as reloadable files, meaning predictions can be served instantly without retraining. The pipeline is modular: each of the four stages (data preparation, feature engineering, modeling, dashboard) is an independent script that can be updated or rerun without affecting the others.

---

## Current Limitations and Next Steps

| Limitation | Recommended Next Step |
|---|---|
| Models do not capture time trends or seasonality | Integrate a time-series layer (monthly enrollment trajectories) |
| Predictions are point estimates with no confidence range | Add prediction intervals to the dashboard for risk-aware planning |
| Instructor assignment uses a simple majority rule | Model co-instruction and instructor transitions more precisely |
| No learner-level behavioral data | Incorporate completion rates and review sentiment when available |

---

## Bottom Line

EduPro Analysis transforms internal transaction and catalog data into a strategic forecasting tool. It tells platform operators which courses are likely to succeed before committing resources, which categories deserve investment, and how pricing and instructor decisions ripple through into enrollment and revenue. The dashboard makes these insights available to any team member — immediately, interactively, and without technical overhead.

---
