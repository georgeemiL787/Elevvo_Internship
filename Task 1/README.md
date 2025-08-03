# Student Performance Prediction Project

## Project Overview
This project aims to predict students' exam scores based on various factors including study hours, attendance, parental involvement, and other academic and personal factors using machine learning techniques.

## Objectives
- Perform data cleaning and exploratory data analysis
- Build a linear regression model to predict exam scores
- Evaluate model performance using various metrics
- Visualize results and insights
- Experiment with polynomial regression and feature engineering

## Dataset
- **Source**: Student Performance Factors (Kaggle)
- **Size**: 6,609 records with 20 features
- **Target Variable**: Exam_Score
- **Features**: Study hours, attendance, parental involvement, access to resources, extracurricular activities, sleep hours, previous scores, motivation level, internet access, tutoring sessions, family income, teacher quality, school type, peer influence, physical activity, learning disabilities, parental education level, distance from home, and gender.

## Project Structure
```
Task 1/
├── StudentPerformanceFactors.csv    # Dataset
├── Task_1.ipynb                     # Main Jupyter notebook
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### 1. Install Dependencies
First, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook
```

### 3. Open the Notebook
Open `Task_1.ipynb` in your browser and run the cells sequentially.

## Project Steps

### Step 1: Setup and Import Libraries
- Import required libraries (numpy, pandas, matplotlib, seaborn, scikit-learn)
- Set up visualization styles

### Step 2: Load and Explore the Dataset
- Load the CSV file
- Display basic information about the dataset
- Check data types and structure

### Step 3: Data Cleaning and Preprocessing
- Check for missing values and handle them
- Remove duplicates if any
- Verify data quality

### Step 4: Exploratory Data Analysis
- Analyze target variable distribution
- Create correlation matrix for numerical features
- Visualize relationships between variables
- Analyze categorical variables' impact on exam scores

### Step 5: Feature Engineering
- Encode categorical variables using Label Encoding
- Create interaction features (e.g., study hours × attendance)
- Prepare features for modeling

### Step 6: Linear Regression Model
- Split data into training and testing sets
- Train linear regression model with feature scaling
- Evaluate model performance using multiple metrics
- Analyze feature importance

### Step 7: Polynomial Regression (Bonus)
- Create polynomial features (degree=2)
- Train polynomial regression model
- Compare performance with linear regression

### Step 8: Visualization of Results
- Plot actual vs predicted values
- Create residuals plots
- Visualize feature importance

### Step 9: Feature Combination Experimentation (Bonus)
- Test different feature combinations
- Compare model performance across combinations
- Identify most predictive feature sets

### Step 10: Conclusions and Insights
- Summarize key findings
- Provide recommendations
- Discuss model limitations

## Key Features

### Data Analysis
- Comprehensive exploratory data analysis
- Correlation analysis between features
- Visualization of data distributions and relationships

### Machine Learning Models
- **Linear Regression**: Baseline model with feature scaling
- **Polynomial Regression**: Advanced model with polynomial features
- **Feature Engineering**: Interaction features and categorical encoding

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²) Score

### Visualizations
- Distribution plots of exam scores
- Correlation heatmaps
- Actual vs predicted scatter plots
- Residual analysis plots
- Feature importance charts

## Expected Outcomes

### Model Performance
- Linear Regression typically achieves R² > 0.7
- Polynomial Regression may show slight improvement
- RMSE around 3-4 points on exam scores

### Key Insights
- Study hours and attendance are primary predictors
- Previous academic performance strongly correlates with exam scores
- Parental involvement and teacher quality significantly impact performance
- Sleep and motivation levels affect academic outcomes

### Feature Importance
Top predictive features usually include:
1. Previous_Scores
2. Hours_Studied
3. Attendance
4. Study_Attendance_Interaction
5. Motivation_Level

## Bonus Features

### Polynomial Regression
- Tests non-linear relationships
- May capture complex feature interactions
- Compares performance with linear model

### Feature Experimentation
- Tests different feature combinations
- Identifies optimal feature subsets
- Provides insights into feature relevance

### Advanced Analysis
- Residual analysis for model diagnostics
- Feature importance ranking
- Model comparison and selection

## Requirements

### Python Version
- Python 3.7 or higher

### Key Libraries
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms
- **jupyter**: Interactive notebook environment

## Running the Project

1. **Clone or download** the project files
2. **Navigate** to the Task 1 directory
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Launch Jupyter**: `jupyter notebook`
5. **Open Task_1.ipynb** and run cells sequentially
6. **Follow the markdown instructions** in each section

## Troubleshooting

### Common Issues
- **ModuleNotFoundError**: Ensure all dependencies are installed
- **Memory issues**: Consider using a subset of data for testing
- **Display issues**: Restart kernel if plots don't appear

### Performance Tips
- Use smaller datasets for initial testing
- Close other applications to free memory
- Consider using Google Colab for cloud execution

## Project Deliverables

1. **Complete Jupyter Notebook** with all analysis steps
2. **Data cleaning and preprocessing** documentation
3. **Model performance evaluation** with metrics
4. **Visualization outputs** and insights
5. **Feature importance analysis**
6. **Model comparison** (linear vs polynomial)
7. **Recommendations and conclusions**

## Learning Objectives

This project covers:
- **Data Science Workflow**: From data loading to model deployment
- **Exploratory Data Analysis**: Understanding data patterns and relationships
- **Feature Engineering**: Creating meaningful features for modeling
- **Machine Learning**: Linear and polynomial regression
- **Model Evaluation**: Multiple performance metrics
- **Data Visualization**: Creating informative plots and charts
- **Statistical Analysis**: Correlation and feature importance

## Next Steps

After completing this project, consider:
- Trying other algorithms (Random Forest, XGBoost)
- Cross-validation for more robust evaluation
- Hyperparameter tuning
- Feature selection techniques
- Deployment of the model as a web application

---

**Note**: This project is designed for educational purposes and demonstrates fundamental machine learning concepts using real-world data. 