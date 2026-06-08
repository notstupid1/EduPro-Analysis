"""
Feature Engineering Module for EduPro Predictive Modeling
Creates and transforms features for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


def create_price_bands():
    """Create price band categories"""

    merged_data['price_band'] = pd.cut(
        merged_data['CoursePrice'],
        bins=[0, 100, 300, float('inf')],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    merged_data['price_band'] = merged_data['price_band'].fillna('low')

def create_duration_buckets():
    """Create duration bucket categories"""

    merged_data['duration_bucket'] = pd.cut(
        merged_data['CourseDuration'],
        bins=[0, 10, 30, float('inf')],
        labels=['short', 'medium', 'long'],
        include_lowest=True
    )
    merged_data['duration_bucket'] = merged_data['duration_bucket'].fillna('short')

def create_rating_tiers():
    """Create rating tier categories for courses and teachers"""
    
    # Course rating tiers
    merged_data['course_rating_tier'] = pd.cut(
        merged_data['CourseRating'],
        bins=[0, 2, 3, 4, float('inf')],
        labels=['poor', 'average', 'good', 'excellent'],
        include_lowest=True
    )
    merged_data['course_rating_tier'] = merged_data['course_rating_tier'].fillna('average')
    
    # Teacher rating tiers
    merged_data['teacher_rating_tier'] = pd.cut(
        merged_data['TeacherRating'],
        bins=[0, 2, 3, 4, float('inf')],
        labels=['poor', 'average', 'good', 'excellent'],
        include_lowest=True
    )
    merged_data['teacher_rating_tier'] = merged_data['teacher_rating_tier'].fillna('average')

def create_experience_buckets():
    """Create teacher experience buckets"""

    merged_data['experience_level'] = pd.cut(
        merged_data['YearsOfExperience'],
        bins=[0, 3, 7, float('inf')],
        labels=['junior', 'mid', 'senior'],
        include_lowest=True
    )
    merged_data['experience_level'] = merged_data['experience_level'].fillna('junior')

def create_level_encoding():
    """Create numeric encoding for course level"""

    level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    merged_data['course_level_encoded'] = merged_data['CourseLevel'].map(level_map)
    merged_data['course_level_encoded'] = merged_data['course_level_encoded'].fillna(1)

def create_category_encoding():
    """Create numeric encoding for course category"""

    category_map = {'Digital Marketing': 1, 'Business': 2, 'Design': 3, 'Programming': 4, 'Marketing': 5, 
                    'Data Science': 6, 'Machine Learning': 7, 'Cybersecurity': 8, 'Finance': 9, 'Project Management': 10,
                    'Artificial Intelligence': 11, 'Web Development': 12}
    merged_data['course_category_encoded'] = merged_data['CourseCategory'].map(category_map)
    merged_data['course_category_encoded'] = merged_data['course_category_encoded'].fillna(1)

def create_expertise_category_match():
    """Create expertise-category match score"""
    
    # Map expertise areas to categories (simplified)
    expertise_category_map = {
        'Cybersecurity': 'Technology',
        'Digital Marketing': 'Business',
        'Data Science': 'Technology',
        'Machine Learning': 'Technology',
        'Business Management': 'Business',
        'Finance': 'Business',
        'Web Development': 'Technology',
        'Mobile Development': 'Technology',
        'Cloud Computing': 'Technology',
        'UI/UX Design': 'Design',
    }
    
    merged_data['teacher_expertise_mapped'] = merged_data['Expertise'].map(expertise_category_map).fillna('Other')
    
    # Create match score (1 if teacher expertise matches course category, 0 otherwise)
    merged_data['expertise_category_match'] = (
        merged_data['teacher_expertise_mapped'] == merged_data['CourseCategory']
    ).astype(int)

def create_category_aggregates():
    """Create category-level aggregate features"""
    global merged_data

    category_stats = merged_data.groupby('CourseCategory').agg({
        'CourseRating': 'mean',
        'CourseID': 'count',
        'enrollment_count': ['mean', 'sum'],
        'total_revenue': ['mean', 'sum']
    }).round(2)
    
    category_stats.columns = ['avg_category_rating', 'category_course_count', 
                               'avg_category_enrollment', 'total_category_enrollment', 'avg_category_revenue', 'total_category_revenue']
    category_stats = category_stats.reset_index()
    
    merged_data = merged_data.merge(category_stats[['CourseCategory', 'avg_category_rating', 
                                              'category_course_count', 'avg_category_enrollment', 'avg_category_revenue']], 
                           on='CourseCategory', how='left')

def create_teacher_workload():
    """Create teacher workload feature (number of courses taught)"""
    global merged_data

    teacher_workload = merged_data.groupby('TeacherID').size().reset_index(name='teacher_courses_count')
    merged_data = merged_data.merge(teacher_workload, left_on='TeacherID', right_on='TeacherID', how='left')
    merged_data['teacher_courses_count'] = merged_data['teacher_courses_count'].fillna(1)


def create_time_features():
    """Create time-based features from transaction dates"""
    
    # Extract month from first transaction
    merged_data['first_transaction_month'] = pd.to_datetime(merged_data['first_transaction_date']).dt.month
    merged_data['first_transaction_month'] = merged_data['first_transaction_month'].fillna(1)

def encode_categorical_features():
    """One-hot encode categorical features"""
    global merged_data

    categorical_cols = ['CourseCategory', 'CourseType', 'price_band', 'duration_bucket', 
                       'experience_level', 'course_rating_tier', 'teacher_rating_tier']
    
    for col in categorical_cols:
        if col in merged_data.columns:
            # One-hot encoding
            dummies = pd.get_dummies(merged_data[col], prefix=col, drop_first=True)
            merged_data = pd.concat([merged_data, dummies], axis=1)


def normalize_numerical_features():
    """Normalize numerical features"""
    
    numerical_cols = ['CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience',
                     'TeacherRating', 'enrollment_count', 'total_revenue', 'revenue_per_enrollment',
                     'course_age_days', 'avg_category_rating', 'avg_category_enrollment',
                     'teacher_courses_count']
    
    numerical_cols = [col for col in numerical_cols if col in merged_data.columns]
    
    # Create normalized versions
    for col in numerical_cols:
        merged_data[f'{col}_normalized'] = scaler.fit_transform(merged_data[[col]])


def remove_correlated_features(correlation_threshold=0.95):
    """Remove highly correlated features"""
    global merged_data
    # Calculate correlation matrix for numerical features
    numerical_cols = merged_data.select_dtypes(include=[np.number]).columns
    corr_matrix = merged_data[numerical_cols].corr().abs()
    # Get upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    if to_drop:
        merged_data = merged_data.drop(columns=to_drop, errors='ignore')
    

def prepare_features():
    """Run full feature engineering pipeline"""
    
    # Store target variables
    target_cols = ['enrollment_count', 'total_revenue', 'CourseID']
    targets = merged_data[target_cols].copy()
    
    create_price_bands()
    create_duration_buckets()
    create_rating_tiers()
    create_experience_buckets()
    create_level_encoding()
    create_category_encoding()
    create_expertise_category_match()
    create_category_aggregates()
    create_teacher_workload()
    create_time_features()
    encode_categorical_features()
    normalize_numerical_features()
    remove_correlated_features()
    
    # Restore target columns
    for col in target_cols:
        if col not in merged_data.columns:
            merged_data[col] = targets[col]
    
    return merged_data

def save_features():
    """Save engineered features"""
    Path('./data/processed').mkdir(parents=True, exist_ok=True)
    
    output_path = './data/processed/features_engineered.csv'
    engineered_data.to_csv(output_path, index=False)

    # Load and merge data
merged_data_path = "./data/processed/merged_data.csv"
merged_data = pd.read_csv(merged_data_path)
scaler = StandardScaler()
label_encoders = {}
engineered_data = prepare_features()
save_features()

"""print(f"\nEngineered dataset shape: {engineered_data.shape}")
print(f"Total features: {len(engineered_data.columns)}")
print(f"\nFeature columns:\n{list(engineered_data.columns)}")"""
