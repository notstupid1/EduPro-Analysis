"""
Modeling Module for EduPro Predictive Forecasting
Builds, trains, and evaluates predictive models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path


class PredictiveModeler:
    """Build and train predictive models for enrollment and revenue forecasting"""
    
    def __init__(self, engineered_data, test_size=0.2, random_state=42):
        self.engineered_data = engineered_data
        self.test_size = test_size
        self.random_state = random_state
        
        # Separate features and targets
        self.X = None
        self.y_enrollment = None
        self.y_revenue = None
        
        # Train-test split
        self.X_train = None
        self.X_test = None
        self.y_train_enrollment = None
        self.y_test_enrollment = None
        self.y_train_revenue = None
        self.y_test_revenue = None
        
        # Models
        self.models = {}
        self.results = {}
        
    def prepare_features(self):
        """Select and prepare features for modeling"""
        
        # Drop columns that shouldn't be used as features
        drop_cols = ['CourseID', 'TeacherID', 'CourseName', 'TeacherName', 'Age',
                     'first_transaction_date', 'last_transaction_date', 'Expertise',
                     'teacher_expertise_mapped', 'expertise_category_match', 
                     'total_revenue', 'Gender',
                     'CourseLevel', 'CourseType', 'CourseCategory', 'category_course_count', 'first_transaction_month']
        
        # Select only numeric columns
        numeric_df = self.engineered_data.select_dtypes(include=['number'])
        feature_cols = [col for col in numeric_df.columns if col not in drop_cols]
        self.X = numeric_df[feature_cols]

        # Target variables
        self.y_enrollment = self.engineered_data['enrollment_count']
        self.y_revenue = self.engineered_data['total_revenue']
        print(feature_cols)
        return self
    
    def split_data(self):
        """Split data into train and test sets"""
        
        self.X_train, self.X_test = train_test_split(
            self.X, test_size=self.test_size, random_state=self.random_state
        )
        
        self.y_train_enrollment = self.y_enrollment[self.X_train.index]
        self.y_test_enrollment = self.y_enrollment[self.X_test.index]
        
        self.y_train_revenue = self.y_revenue[self.X_train.index]
        self.y_test_revenue = self.y_revenue[self.X_test.index]
        
        return self
    
    def build_baseline_models(self):
        """Build baseline models"""
        
        # Linear Regression - Enrollment
        lr_enrollment = LinearRegression()
        lr_enrollment.fit(self.X_train, self.y_train_enrollment)
        self.models['LinearRegression_Enrollment'] = lr_enrollment
        
        # Linear Regression - Revenue
        lr_revenue = LinearRegression()
        lr_revenue.fit(self.X_train, self.y_train_revenue)
        self.models['LinearRegression_Revenue'] = lr_revenue
        
        # Ridge Regression - Enrollment
        ridge_enrollment = Ridge(alpha=1.0)
        ridge_enrollment.fit(self.X_train, self.y_train_enrollment)
        self.models['Ridge_Enrollment'] = ridge_enrollment
        
        # Ridge Regression - Revenue
        ridge_revenue = Ridge(alpha=1.0)
        ridge_revenue.fit(self.X_train, self.y_train_revenue)
        self.models['Ridge_Revenue'] = ridge_revenue
        
        # Lasso Regression - Enrollment
        lasso_enrollment = Lasso(alpha=0.1)
        lasso_enrollment.fit(self.X_train, self.y_train_enrollment)
        self.models['Lasso_Enrollment'] = lasso_enrollment
        
        # Lasso Regression - Revenue
        lasso_revenue = Lasso(alpha=0.1)
        lasso_revenue.fit(self.X_train, self.y_train_revenue)
        self.models['Lasso_Revenue'] = lasso_revenue
        
        return self
    
    def build_advanced_models(self):
        """Build advanced models"""
        
        # Random Forest - Enrollment
        rf_enrollment = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1)
        rf_enrollment.fit(self.X_train, self.y_train_enrollment)
        self.models['RandomForest_Enrollment'] = rf_enrollment
        
        # Random Forest - Revenue
        rf_revenue = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1)
        rf_revenue.fit(self.X_train, self.y_train_revenue)
        self.models['RandomForest_Revenue'] = rf_revenue
        
        # Gradient Boosting - Enrollment
        gb_enrollment = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state)
        gb_enrollment.fit(self.X_train, self.y_train_enrollment)
        self.models['GradientBoosting_Enrollment'] = gb_enrollment
        
        # Gradient Boosting - Revenue
        gb_revenue = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state)
        gb_revenue.fit(self.X_train, self.y_train_revenue)
        self.models['GradientBoosting_Revenue'] = gb_revenue
        
        return self
    
    def evaluate_models(self):
        """Evaluate all models and print results"""
        
        for model_name, model in self.models.items():
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Determine if enrollment or revenue
            if 'Enrollment' in model_name:
                y_train_true = self.y_train_enrollment
                y_test_true = self.y_test_enrollment
            else:
                y_train_true = self.y_train_revenue
                y_test_true = self.y_test_revenue
            
            # Metrics
            mae_train = mean_absolute_error(y_train_true, y_pred_train)
            mae_test = mean_absolute_error(y_test_true, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test_true, y_pred_test))
            r2_train = r2_score(y_train_true, y_pred_train)
            r2_test = r2_score(y_test_true, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, y_train_true, cv=5, scoring='r2')
            
            self.results[model_name] = {
                'mae_train': mae_train,
                'mae_test': mae_test,
                'rmse_test': rmse_test,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"\n{model_name}")
            print("-" * 50)
            print(f"  Train MAE:        {mae_train:.4f}")
            print(f"  Test MAE:         {mae_test:.4f}")
            print(f"  Test RMSE:        {rmse_test:.4f}")
            print(f"  Train R²:         {r2_train:.4f}")
            print(f"  Test R²:          {r2_test:.4f}")
            print(f"  CV R² (5-fold):   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        model = self.models.get(model_name)
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_models(self):
        """Save trained models"""
        Path('./models').mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f'./models/{model_name}.pkl'
            joblib.dump(model, model_path)
        
        return self
    
    def train_all(self):
        """Run full training pipeline"""
        
        self.prepare_features()
        self.split_data()
        self.build_baseline_models()
        self.build_advanced_models()
        self.evaluate_models()
        
        return self


if __name__ == "__main__":
    
    # Load and prepare data
    engineered_data_path = "./data/processed/features_engineered.csv"
    engineered_data = pd.read_csv(engineered_data_path)
    
    # Train models
    modeler = PredictiveModeler(engineered_data)
    modeler.train_all()
    modeler.save_models()
    
    # Feature importance
    for model_name in ['RandomForest_Enrollment', 'RandomForest_Revenue', 'GradientBoosting_Enrollment', 'GradientBoosting_Revenue']:
        print(f"\n{model_name} - Top 10 Features:")
        importance = modeler.get_feature_importance(model_name)
        if importance is not None:
            print(importance)