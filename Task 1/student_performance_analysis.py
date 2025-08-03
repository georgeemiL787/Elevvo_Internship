#!/usr/bin/env python3
"""
Student Performance Prediction Project
=====================================

This script performs a complete analysis of student performance data including:
- Data loading and exploration
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Linear and polynomial regression modeling
- Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def main():
    print("=== STUDENT PERFORMANCE PREDICTION PROJECT ===\n")
    
    # Step 1: Load and Explore the Dataset
    print("Step 1: Loading and exploring the dataset...")
    df = load_and_explore_data()
    
    # Step 2: Check Unique Values
    print("\nStep 2: Checking unique values...")
    check_unique_values(df)
    
    # Step 3: Data Cleaning
    print("\nStep 3: Data cleaning and preprocessing...")
    df_clean = clean_data(df)
    
    # Step 4: Exploratory Data Analysis
    print("\nStep 4: Exploratory data analysis...")
    perform_eda(df_clean)
    
    # Step 5: Feature Engineering
    print("\nStep 5: Feature engineering...")
    df_processed = engineer_features(df_clean)
    
    # Step 6: Model Training and Evaluation
    print("\nStep 6: Model training and evaluation...")
    train_and_evaluate_models(df_processed)
    
    print("\n=== PROJECT COMPLETED SUCCESSFULLY! ===")

def load_and_explore_data():
    """Load the dataset and display basic information"""
    try:
        # Load the dataset
        df = pd.read_csv('StudentPerformanceFactors.csv')
        
        # Display basic information
        print(f"Dataset Shape: {df.shape}")
        print(f"Number of records: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nBasic Statistics:")
        print(df.describe())
        
        return df
        
    except FileNotFoundError:
        print("Error: StudentPerformanceFactors.csv not found!")
        return None

def check_unique_values(df):
    """Check unique values for each column in the dataset"""
    print("Checking unique values for each column:\n")
    
    for column in df.columns:
        unique_count = df[column].nunique()
        total_count = len(df[column])
        
        print(f"Column: {column}")
        print(f"  - Total values: {total_count}")
        print(f"  - Unique values: {unique_count}")
        print(f"  - Missing values: {df[column].isnull().sum()}")
        
        # Show unique values for categorical columns or columns with few unique values
        if unique_count <= 20 or df[column].dtype == 'object':
            unique_vals = df[column].value_counts()
            print(f"  - Unique values and counts:")
            for val, count in unique_vals.items():
                print(f"    {val}: {count}")
        else:
            print(f"  - Range: {df[column].min()} to {df[column].max()}")
            print(f"  - Sample unique values: {list(df[column].unique()[:5])}")
        
        print("-" * 50)

def clean_data(df):
    """Clean and preprocess the data"""
    df_clean = df.copy()
    
    # Check for missing values
    print("Missing values per column:")
    print(df_clean.isnull().sum())
    
    # Check for duplicates
    print(f"\nNumber of duplicate rows: {df_clean.duplicated().sum()}")
    
    # Handle missing values if any
    if df_clean.isnull().sum().sum() > 0:
        print("\nHandling missing values...")
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        print("Missing values handled!")
    else:
        print("\nNo missing values found!")
    
    return df_clean

def perform_eda(df):
    """Perform exploratory data analysis"""
    print("Performing exploratory data analysis...")
    
    # Analyze target variable
    print(f"\nTarget Variable (Exam_Score) Statistics:")
    print(f"Mean: {df['Exam_Score'].mean():.2f}")
    print(f"Median: {df['Exam_Score'].median():.2f}")
    print(f"Std: {df['Exam_Score'].std():.2f}")
    print(f"Min: {df['Exam_Score'].min():.2f}")
    print(f"Max: {df['Exam_Score'].max():.2f}")
    
    # Correlation analysis for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    print(f"\nTop correlations with Exam Score:")
    exam_correlations = correlation_matrix['Exam_Score'].sort_values(ascending=False)
    print(exam_correlations)
    
    # Analyze categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nAverage Exam Scores by Categorical Variables:")
    for col in categorical_cols:
        print(f"\n{col}:")
        avg_scores = df.groupby(col)['Exam_Score'].agg(['mean', 'count']).round(2)
        print(avg_scores)

def engineer_features(df):
    """Engineer features for modeling"""
    print("Engineering features...")
    
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Create interaction features
    df_processed['Study_Attendance_Interaction'] = df_processed['Hours_Studied'] * df_processed['Attendance']
    df_processed['Study_PreviousScore_Interaction'] = df_processed['Hours_Studied'] * df_processed['Previous_Scores']
    
    print("Feature engineering completed!")
    return df_processed

def train_and_evaluate_models(df_processed):
    """Train and evaluate machine learning models"""
    print("Training and evaluating models...")
    
    # Prepare features for modeling
    feature_cols = [col for col in df_processed.columns if col not in ['Exam_Score'] + 
                   list(df_processed.select_dtypes(include=['object']).columns)]
    
    X = df_processed[feature_cols]
    y = df_processed['Exam_Score']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Linear Regression
    print("\nTraining Linear Regression model...")
    linear_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    linear_pipeline.fit(X_train, y_train)
    y_pred_linear = linear_pipeline.predict(X_test)
    
    # Evaluate Linear Regression
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    
    print("Linear Regression Results:")
    print(f"Mean Squared Error: {mse_linear:.2f}")
    print(f"Root Mean Squared Error: {rmse_linear:.2f}")
    print(f"Mean Absolute Error: {mae_linear:.2f}")
    print(f"R² Score: {r2_linear:.4f}")
    
    # Polynomial Regression
    print("\nTraining Polynomial Regression model...")
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    poly_pipeline.fit(X_train, y_train)
    y_pred_poly = poly_pipeline.predict(X_test)
    
    # Evaluate Polynomial Regression
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    print("Polynomial Regression Results:")
    print(f"Mean Squared Error: {mse_poly:.2f}")
    print(f"Root Mean Squared Error: {rmse_poly:.2f}")
    print(f"Mean Absolute Error: {mae_poly:.2f}")
    print(f"R² Score: {r2_poly:.4f}")
    
    # Model Comparison
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Linear Regression': [mse_linear, rmse_linear, mae_linear, r2_linear],
        'Polynomial Regression': [mse_poly, rmse_poly, mae_poly, r2_poly]
    })
    print(comparison_df.round(4))
    
    # Feature importance for linear regression
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': linear_pipeline.named_steps['regressor'].coef_
    })
    feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nTop 10 Most Important Features (Linear Regression):")
    print(feature_importance.head(10))
    
    # Step 8: Visualization of Results
    print("\nStep 8: Creating visualizations...")
    visualize_results(y_test, y_pred_linear, y_pred_poly, feature_importance, None)
    
    # Step 9: Feature Combination Experimentation (Bonus)
    print("\nStep 9: Feature combination experimentation...")
    results_df = feature_combination_experimentation(X_train, X_test, y_train, y_test, feature_cols)
    
    # Final visualization with feature combination results
    print("\nFinal visualization with feature combination results...")
    visualize_results(y_test, y_pred_linear, y_pred_poly, feature_importance, results_df)

def experiment_with_feature_combinations(X_train, X_test, y_train, y_test, feature_cols):
    """Experiment with different feature combinations"""
    
    def evaluate_feature_combination(features, X_train, X_test, y_train, y_test):
        """Evaluate model performance with given feature combination"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        pipeline.fit(X_train[features], y_train)
        y_pred = pipeline.predict(X_test[features])
        
        return {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
    
    # Define different feature combinations
    feature_combinations = {
        'All Features': feature_cols,
        'Core Academic': ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions'],
        'Study Focus': ['Hours_Studied', 'Sleep_Hours', 'Motivation_Level', 'Previous_Scores'],
        'Support Factors': ['Parental_Involvement_encoded', 'Access_to_Resources_encoded', 
                           'Teacher_Quality_encoded', 'Family_Income_encoded'],
        'Personal Factors': ['Sleep_Hours', 'Physical_Activity', 'Motivation_Level', 
                            'Learning_Disabilities_encoded', 'Gender_encoded']
    }
    
    # Evaluate each combination
    results = {}
    for name, features in feature_combinations.items():
        # Check if all features exist in the dataset
        available_features = [f for f in features if f in feature_cols]
        if available_features:
            results[name] = evaluate_feature_combination(available_features, X_train, X_test, y_train, y_test)
    
    # Display results
    results_df = pd.DataFrame(results).T
    print("Feature Combination Comparison:")
    print(results_df.round(4))
    
    return results_df

def visualize_results(y_test, y_pred_linear, y_pred_poly, feature_importance, results_df):
    """Step 8: Visualization of Results"""
    print("\n" + "="*60)
    print("STEP 8: VISUALIZATION OF RESULTS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Student Performance Prediction - Model Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted - Linear Regression
    axes[0, 0].scatter(y_test, y_pred_linear, alpha=0.6, color='blue', s=30)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Exam Score')
    axes[0, 0].set_ylabel('Predicted Exam Score')
    axes[0, 0].set_title('Linear Regression: Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² score to plot
    r2_linear = r2_score(y_test, y_pred_linear)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2_linear:.4f}', 
                    transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontweight='bold')
    
    # 2. Actual vs Predicted - Polynomial Regression
    axes[0, 1].scatter(y_test, y_pred_poly, alpha=0.6, color='green', s=30)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Exam Score')
    axes[0, 1].set_ylabel('Predicted Exam Score')
    axes[0, 1].set_title('Polynomial Regression: Actual vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add R² score to plot
    r2_poly = r2_score(y_test, y_pred_poly)
    axes[0, 1].text(0.05, 0.95, f'R² = {r2_poly:.4f}', 
                    transform=axes[0, 1].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontweight='bold')
    
    # 3. Residuals Plot
    residuals_linear = y_test - y_pred_linear
    residuals_poly = y_test - y_pred_poly
    
    axes[0, 2].scatter(y_pred_linear, residuals_linear, alpha=0.6, color='blue', s=30, label='Linear')
    axes[0, 2].scatter(y_pred_poly, residuals_poly, alpha=0.6, color='green', s=30, label='Polynomial')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 2].set_xlabel('Predicted Exam Score')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residuals Plot')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Feature Importance
    top_features = feature_importance.head(10)
    y_pos = np.arange(len(top_features))
    colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
    
    axes[1, 0].barh(y_pos, top_features['Coefficient'], color=colors, alpha=0.7)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(top_features['Feature'])
    axes[1, 0].set_xlabel('Coefficient Value')
    axes[1, 0].set_title('Top 10 Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add coefficient values to bars
    for i, v in enumerate(top_features['Coefficient']):
        axes[1, 0].text(v + (0.01 if v > 0 else -0.01), i, f'{v:.3f}', 
                       va='center', fontweight='bold', fontsize=8)
    
    # 5. Model Performance Comparison
    metrics = ['R²', 'RMSE', 'MAE']
    linear_scores = [r2_linear, np.sqrt(mean_squared_error(y_test, y_pred_linear)), 
                    mean_absolute_error(y_test, y_pred_linear)]
    poly_scores = [r2_poly, np.sqrt(mean_squared_error(y_test, y_pred_poly)), 
                  mean_absolute_error(y_test, y_pred_poly)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, linear_scores, width, label='Linear Regression', alpha=0.8)
    axes[1, 1].bar(x + width/2, poly_scores, width, label='Polynomial Regression', alpha=0.8)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature Combination Performance
    if results_df is not None and not results_df.empty:
        feature_combinations = results_df.index
        r2_scores = results_df['R²'].values
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_combinations)))
        bars = axes[1, 2].bar(feature_combinations, r2_scores, color=colors, alpha=0.8)
        axes[1, 2].set_xlabel('Feature Combinations')
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title('Feature Combination Performance')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add R² values on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'No feature combination data available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Feature Combination Performance')
    
    plt.tight_layout()
    plt.show()
    
    print("Visualizations completed and displayed!")

def feature_combination_experimentation(X_train, X_test, y_train, y_test, feature_cols):
    """Step 9: Feature Combination Experimentation (Bonus)"""
    print("\n" + "="*60)
    print("STEP 9: FEATURE COMBINATION EXPERIMENTATION (BONUS)")
    print("="*60)
    
    def evaluate_feature_combination(features, X_train, X_test, y_train, y_test):
        """Evaluate model performance with given feature combination"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        pipeline.fit(X_train[features], y_train)
        y_pred = pipeline.predict(X_test[features])
        
        return {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
    
    # Define different feature combinations
    feature_combinations = {
        'All Features': feature_cols,
        'Core Academic': ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions'],
        'Study Focus': ['Hours_Studied', 'Sleep_Hours', 'Motivation_Level', 'Previous_Scores'],
        'Support Factors': ['Parental_Involvement_encoded', 'Access_to_Resources_encoded', 
                           'Teacher_Quality_encoded', 'Family_Income_encoded'],
        'Personal Factors': ['Sleep_Hours', 'Physical_Activity', 'Motivation_Level', 
                            'Learning_Disabilities_encoded', 'Gender_encoded'],
        'Study + Support': ['Hours_Studied', 'Attendance', 'Parental_Involvement_encoded', 
                           'Teacher_Quality_encoded', 'Access_to_Resources_encoded'],
        'Performance History': ['Previous_Scores', 'Attendance', 'Tutoring_Sessions'],
        'Lifestyle Factors': ['Sleep_Hours', 'Physical_Activity', 'Motivation_Level', 
                             'Extracurricular_Activities_encoded']
    }
    
    print("Testing different feature combinations...")
    
    # Evaluate each combination
    results = {}
    for name, features in feature_combinations.items():
        # Check if all features exist in the dataset
        available_features = [f for f in features if f in feature_cols]
        if available_features:
            print(f"Testing {name} ({len(available_features)} features)...")
            results[name] = evaluate_feature_combination(available_features, X_train, X_test, y_train, y_test)
        else:
            print(f"Skipping {name} - features not available")
    
    # Display results
    results_df = pd.DataFrame(results).T
    print("\nFeature Combination Comparison:")
    print(results_df.round(4))
    
    # Find best performing combination
    best_r2 = results_df['R²'].idxmax()
    best_rmse = results_df['RMSE'].idxmin()
    
    print(f"\nBest R² Score: {best_r2} ({results_df.loc[best_r2, 'R²']:.4f})")
    print(f"Best RMSE Score: {best_rmse} ({results_df.loc[best_rmse, 'RMSE']:.4f})")
    
    # Create visualization for feature combinations
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² Scores
    r2_scores = results_df['R²'].sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(r2_scores)))
    bars1 = ax1.barh(range(len(r2_scores)), r2_scores, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(r2_scores)))
    ax1.set_yticklabels(r2_scores.index)
    ax1.set_xlabel('R² Score')
    ax1.set_title('Feature Combinations by R² Score')
    ax1.grid(True, alpha=0.3)
    
    # Add R² values on bars
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        ax1.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')
    
    # RMSE Scores
    rmse_scores = results_df['RMSE'].sort_values(ascending=False)
    colors = plt.cm.plasma(np.linspace(0, 1, len(rmse_scores)))
    bars2 = ax2.barh(range(len(rmse_scores)), rmse_scores, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(rmse_scores)))
    ax2.set_yticklabels(rmse_scores.index)
    ax2.set_xlabel('RMSE Score')
    ax2.set_title('Feature Combinations by RMSE Score')
    ax2.grid(True, alpha=0.3)
    
    # Add RMSE values on bars
    for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
        ax2.text(score + 0.1, i, f'{score:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nFeature combination experimentation completed!")
    return results_df

if __name__ == "__main__":
    main() 