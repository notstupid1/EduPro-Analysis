"""
Streamlit Web Application for EduPro Demand & Revenue Forecasting
Interactive dashboard for predictive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="EduPro Demand & Revenue Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card { 
        padding: 1rem; 
        background-color: #f0f2f6; 
        border-radius: 0.5rem; 
        margin: 0.5rem 0;
    }
    .prediction-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare data"""

    merged_data = pd.read_csv('./data/processed/merged_data.csv')
    
    engineered_data = pd.read_csv('./data/processed/features_engineered.csv')
    
    return merged_data, engineered_data

@st.cache_resource
def load_models():
    """Load trained models"""
    models_dir = Path('./models')
    
    models = {}
    model_files = list(models_dir.glob('*.pkl'))
    
    for model_file in model_files:
        model_name = model_file.stem
        try:
            models[model_name] = joblib.load(model_file)
        except:
            pass
    
    return models

def main():
    """Main Streamlit app"""
    
    # Sidebar
    st.sidebar.title("🎓 EduPro Analytics")
    page = st.sidebar.radio(
        "Select Dashboard",
        ["🏠 Overview", "📈 Demand Prediction", "💰 Revenue Forecasting", 
         "🔍 Feature Importance", "📊 Category Analysis", "💡 Insights"]
    )
    
    # Load data
    try:
        merged_data, engineered_data = load_data()
        models = load_models()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the data preparation pipeline first:")
        st.code("python data_preparation.py && python feature_engineering.py && python modeling.py")
        return
    
    # PAGE 1: OVERVIEW
    if page == "🏠 Overview":
        st.title("📊 EduPro Demand & Revenue Forecasting Dashboard")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Courses", len(merged_data))
        with col2:
            st.metric("Total Enrollments", int(merged_data['enrollment_count'].sum()))
        with col3:
            st.metric("Total Revenue", f"${merged_data['total_revenue'].sum():,.2f}")
        with col4:
            st.metric("Avg Course Rating", f"{merged_data['CourseRating'].mean():.2f}/5")
        
        st.markdown("---")
        st.subheader("📈 Key Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top courses by enrollment
            top_courses = merged_data.nlargest(10, 'enrollment_count')[['CourseName', 'enrollment_count', 'total_revenue']]
            fig = px.bar(top_courses, x='enrollment_count', y='CourseName', orientation='h',
                        title="Top 10 Courses by Enrollment", color='enrollment_count',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue by category
            revenue_by_cat = merged_data.groupby('CourseCategory')['total_revenue'].sum().sort_values(ascending=False)
            fig = px.pie(values=revenue_by_cat.values, names=revenue_by_cat.index,
                        title="Revenue Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(merged_data, x='CoursePrice', nbins=20, title="Course Price Distribution",
                             color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(merged_data, x='enrollment_count', nbins=20, title="Enrollment Distribution",
                             color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 2: DEMAND PREDICTION
    elif page == "📈 Demand Prediction":
        st.title("📈 Course Demand Prediction")
        st.markdown("---")
        
        st.subheader("🔮 Predict Course Enrollment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            course_price = st.slider("Course Price ($)", 0.0, 1000.0, 300.0, step=10.0)
        with col2:
            course_duration = st.slider("Course Duration (hours)", 1.0, 50.0, 20.0, step=1.0)
        with col3:
            course_level = st.selectbox("Course Level", ["Beginner", "Intermediate", "Advanced"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            teacher_experience = st.slider("Teacher Experience (years)", 0, 20, 5)
        with col2:
            teacher_rating = st.slider("Teacher Rating (out of 5)", 1.0, 5.0, 3.5, step=0.1)
        with col3:
            course_category = st.selectbox("Course Category", merged_data['CourseCategory'].unique())
        
        course_rating = st.slider("Expected Course Rating (out of 5)", 1.0, 5.0, 3.5, step=0.1)

        level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        category_map = {'Digital Marketing': 1, 'Business': 2, 'Design': 3, 'Programming': 4, 'Marketing': 5, 
                         'Data Science': 6, 'Machine Learning': 7, 'Cybersecurity': 8, 'Finance': 9, 'Project Management': 10,
                         'Artificial Intelligence': 11, 'Web Development': 12}

        if st.button("🎯 Predict Enrollment"):
            st.info("Predictions loaded from training data")
            for model_name, model in models.items():
                avg_category_enrollment, avg_category_revenue, avg_category_rating = engineered_data.filter(items=['avg_category_enrollment', 'avg_category_revenue', 'avg_category_rating']).loc[engineered_data['CourseCategory'] == course_category].iloc[0]
                if model_name in ['RandomForest_Enrollment', 'GradientBoosting_Enrollment', 'RandomForest_Revenue', 'GradientBoosting_Revenue']:
                    if 'Enrollment' in model_name:
                        st.write(f"**{model_name}**: {int(model.predict([[course_price, course_duration, level_map.get(course_level, 1), 
                                                                          teacher_experience, teacher_rating, category_map.get(course_category, 1), 
                                                                          course_rating, 
                                                                          avg_category_enrollment, 
                                                                          avg_category_revenue, 
                                                                          avg_category_rating, 
                                                                          course_rating]])[0])} enrollments")
                    else:
                        st.write(f"**{model_name}**: ${model.predict([[course_price, course_duration, level_map.get(course_level, 1), 
                                                                       teacher_experience, teacher_rating, category_map.get(course_category, 1), 
                                                                       course_rating, 
                                                                       avg_category_enrollment, 
                                                                       avg_category_revenue, 
                                                                       avg_category_rating, 
                                                                       course_rating]])[0]:,.2f} revenue")
            
            # Show similar courses
            st.subheader("📌 Similar Courses")
            similar = merged_data[
                (merged_data['CourseLevel'] == course_level) & 
                (merged_data['CourseCategory'] == course_category)
            ].nlargest(5, 'enrollment_count')[['CourseName', 'enrollment_count', 'CoursePrice', 'CourseRating']]
            
            st.dataframe(similar, use_container_width=True)
    
    # PAGE 3: REVENUE FORECASTING
    elif page == "💰 Revenue Forecasting":
        st.title("💰 Revenue Forecasting")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💵 Revenue by Category")
            revenue_by_cat = merged_data.groupby('CourseCategory').agg({
                'total_revenue': 'sum',
                'enrollment_count': 'sum'
            }).sort_values('total_revenue', ascending=False)
            
            fig = px.bar(revenue_by_cat.reset_index(), x='CourseCategory', y='total_revenue',
                        title="Total Revenue by Category", color='total_revenue',
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Revenue per Enrollment")
            merged_data['rev_per_enrollment'] = merged_data['total_revenue'] / (merged_data['enrollment_count'] + 1)
            fig = px.box(merged_data, x='CourseCategory', y='rev_per_enrollment',
                        title="Revenue per Enrollment by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: FEATURE IMPORTANCE
    elif page == "🔍 Feature Importance":
        st.title("🔍 Feature Importance Analysis")
        st.markdown("---")
        
        if not models:
            st.warning("No trained models found. Please train models first.")
            return
        
        model_name = st.selectbox(
            "Select Model",
            [m for m in models.keys() if 'RandomForest' in m or 'GradientBoosting' in m]
        )
        
        st.info(f"Showing feature importance for: {model_name}")

        drop_cols = ['CourseID', 'TeacherID', 'CourseName', 'TeacherName', 'Age',
                     'first_transaction_date', 'last_transaction_date', 'Expertise',
                     'teacher_expertise_mapped', 'expertise_category_match', 
                     'total_revenue', 'Gender',
                     'CourseLevel', 'CourseType', 'CourseCategory', 'category_course_count', 'first_transaction_month']
        
        model = models.get(model_name)
        if hasattr(model, 'feature_importances_'):
            numeric_df = engineered_data.select_dtypes(include=['number'])
            features = [col for col in numeric_df.columns if col not in drop_cols]
            #features = engineered_data.columns
            importances = model.feature_importances_

            if len(features) != len(importances):
                st.error("Mismatch between feature and importance counts.")

            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title=f"Top Features - {model_name}",
                        color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 5: CATEGORY ANALYSIS
    elif page == "📊 Category Analysis":
        st.title("📊 Category-Level Analysis")
        st.markdown("---")
        
        # Category statistics
        category_stats = merged_data.groupby('CourseCategory').agg({
            'CourseID': 'count',
            'enrollment_count': ['sum', 'mean'],
            'total_revenue': ['sum', 'mean'],
            'CourseRating': 'mean',
            'TeacherRating': 'mean'
        }).round(2)
        
        category_stats.columns = ['Courses', 'Total Enrollments', 'Avg Enrollment', 
                                 'Total Revenue', 'Avg Revenue', 'Avg Course Rating', 'Avg Teacher Rating']
        
        st.dataframe(category_stats, use_container_width=True)
        
        # Interactive category selection
        selected_category = st.selectbox("Select Category for Details", merged_data['CourseCategory'].unique())
        
        category_courses = merged_data[merged_data['CourseCategory'] == selected_category].sort_values('enrollment_count', ascending=False)
        
        st.subheader(f"Courses in {selected_category}")
        st.dataframe(
            category_courses[['CourseName', 'CourseLevel', 'CoursePrice', 'enrollment_count', 'total_revenue', 'CourseRating']],
            use_container_width=True
        )
    
    # PAGE 6: INSIGHTS
    elif page == "💡 Insights":
        st.title("💡 Key Insights & Recommendations")
        st.markdown("---")
        
        with st.expander("📌 Top Performing Courses"):
            top_5 = merged_data.nlargest(5, 'enrollment_count')[['CourseName', 'CourseCategory', 'enrollment_count', 'total_revenue']]
            for idx, row in top_5.iterrows():
                st.write(f"**{row['CourseName']}** - {row['enrollment_count']} enrollments, ${row['total_revenue']:,.2f} revenue")
        
        with st.expander("🎓 Category Performance"):
            for cat in merged_data['CourseCategory'].unique():
                cat_data = merged_data[merged_data['CourseCategory'] == cat]
                st.write(f"**{cat}**: {len(cat_data)} courses, {int(cat_data['enrollment_count'].sum())} total enrollments")
        
        with st.expander("⭐ Instructor Quality"):
            top_teachers = merged_data.groupby('TeacherName').agg({
                'enrollment_count': 'sum',
                'TeacherRating': 'first'
            }).sort_values('enrollment_count', ascending=False).head(5)
            st.dataframe(top_teachers)
        
        with st.expander("💰 Pricing Insights"):
            price_analysis = engineered_data.groupby('price_band').agg({
                'enrollment_count': ['sum', 'mean'],
                'total_revenue': 'mean'
            }).round(2)
            st.dataframe(price_analysis)

if __name__ == "__main__":
    main()
