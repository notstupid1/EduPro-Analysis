"""
Data Preparation Module for EduPro Predictive Modeling
Handles loading, merging, and aggregating data from multiple sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_data(): 
    courses = pd.read_excel(excel_path, sheet_name='Courses')
    teachers = pd.read_excel(excel_path, sheet_name='Teachers')
    transactions = pd.read_excel(excel_path, sheet_name='Transactions')
    users = pd.read_excel(excel_path, sheet_name='Users')

    return courses, teachers, transactions, users
    
def aggregate_course_metrics():
    """Aggregate transactions at course level"""
    
    # Group transactions by course
    course_agg = transactions.groupby('CourseID').agg({
        'Amount': ['sum', 'mean', 'count'],
        'TransactionDate': ['min', 'max']
    }).round(2)
    
    course_agg.columns = ['total_revenue', 'revenue_per_enrollment', 
                          'enrollment_count', 'first_transaction_date', 'last_transaction_date']
    course_agg = course_agg.reset_index()
    
    # Calculate course age in days
    course_agg['course_age_days'] = (
        (course_agg['last_transaction_date'] - course_agg['first_transaction_date']).dt.days
    )
    
    return course_agg
    
def merge_course_data():
    """Merge courses with aggregated metrics and teacher info"""
    
    # Get aggregated course metrics
    course_agg = aggregate_course_metrics()
    
    # Merge with courses metadata
    merged = courses.merge(course_agg, on='CourseID', how='left')

    # Select the teacher most frequently associated with the course
    teacher_assignments = transactions.groupby('CourseID')['TeacherID'].value_counts().reset_index(name='count')
    teacher_assignments = teacher_assignments.sort_values('count', ascending=False).drop_duplicates('CourseID')
    teacher_assignments = teacher_assignments[['CourseID', 'TeacherID']]
    
    merged = merged.merge(teacher_assignments, on='CourseID', how='left')
    
    # Merge with teacher information
    merged = merged.merge(
        teachers,
        on='TeacherID',
        how='left'
    )
    
    # Handle null values for courses with no transactions
    merged['enrollment_count'] = merged['enrollment_count'].fillna(0).astype(int)
    merged['total_revenue'] = merged['total_revenue'].fillna(0.0)
    merged['revenue_per_enrollment'] = merged['revenue_per_enrollment'].fillna(0.0)
    merged['course_age_days'] = merged['course_age_days'].fillna(0).astype(int)
    
    return merged
    
def save_processed_data():
    """Save processed data to CSV"""
    Path('./data/processed').mkdir(parents=True, exist_ok=True)
    
    output_path = './data/processed/merged_data.csv'
    merged_data.to_csv(output_path, index=False)

def prepare(save=True):
    """Run full preparation pipeline"""
    if save:
        save_processed_data()
    return merged_data


"""def print_data_summary(df):
    """#Print summary statistics of merged data
"""
    print("\n" + "="*80)
    print("MERGED DATA SUMMARY")
    print("="*80)
    print(f"\nTotal Courses: {len(df)}")
    print(f"\nEnrollment Statistics:")
    print(df['enrollment_count'].describe().round(2))
    print(f"\nRevenue Statistics ($):")
    print(df['total_revenue'].describe().round(2))
    print(f"\nCourse Rating Statistics:")
    print(df['CourseRating'].describe().round(2))
    print(f"\nTeacher Rating Statistics:")
    print(df['TeacherRating'].describe().round(2))
    print(f"\nMissing Values:\n{df.isnull().sum()}\n")"""

excel_path = "./data/raw/EduPro Online Platform.xlsx"
courses, teachers, transactions, users = load_data()
merged_data = merge_course_data()

merged_df = prepare(save=True)
    
#print_data_summary(merged_df)
#print(f"\nColumns in merged dataset:\n{list(merged_df.columns)}")

