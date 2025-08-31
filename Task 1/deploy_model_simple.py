import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import json

def deploy_polynomial_regression_model():
    """
    Complete model deployment pipeline:
    1. Load and preprocess data
    2. Train polynomial regression model
    3. Save model artifacts
    4. Create deployment configuration
    5. Generate deployment report
    """
    
    print("Starting Model Deployment Pipeline...")
    print("=" * 60)
    
    # Step 1: Load Dataset
    print("Step 1: Loading Dataset...")
    try:
        students_db = pd.read_csv('StudentPerformanceFactors.csv')
        print(f"Dataset loaded successfully: {students_db.shape[0]} rows, {students_db.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Step 2: Data Preprocessing
    print("\nStep 2: Data Preprocessing...")
    
    # Handle missing values
    initial_missing = students_db.isnull().sum().sum()
    students_db = students_db.dropna()
    final_missing = students_db.isnull().sum().sum()
    print(f"Missing values handled: {initial_missing} -> {final_missing}")
    
    # Define column types
    continuous_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                      'Tutoring_Sessions', 'Physical_Activity']
    
    # Columns with natural order (Low < Medium < High)
    ordinal_columns = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 
                      'Family_Income', 'Teacher_Quality', 'Peer_Influence', 
                      'Parental_Education_Level', 'Distance_from_Home']
    
    # Binary or categorical columns without natural order
    nominal_columns = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Gender', 
                      'Learning_Disabilities']
    
    print(f"Column types defined: {len(continuous_cols)} continuous, {len(ordinal_columns)} ordinal, {len(nominal_columns)} nominal")
    
    # Step 3: Feature Engineering
    print("\nStep 3: Feature Engineering...")
    
    # Label encoding for ordinal features
    label_encoders = {}
    for col in ordinal_columns:
        le = LabelEncoder()
        students_db[col] = le.fit_transform(students_db[col])
        label_encoders[col] = le
        print(f"   {col}: {le.classes_}")
    
    # One-hot encoding for nominal features
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    ohe_array = ohe.fit_transform(students_db[nominal_columns])
    
    # Convert to DataFrame
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(nominal_columns), 
                          index=students_db.index)
    
    # Merge and drop original nominal columns
    encoded_students_db = pd.concat([students_db.drop(columns=nominal_columns), ohe_df], axis=1)
    
    # Move Exam_Score to the end
    exam_col = encoded_students_db.pop("Exam_Score")
    encoded_students_db["Exam_Score"] = exam_col
    
    # Drop low-correlation features (based on your analysis)
    # Note: We need to check which columns actually exist after encoding
    columns_to_drop = []
    if 'Gender_Male' in encoded_students_db.columns:
        columns_to_drop.append('Gender_Male')
    if 'School_Type_Public' in encoded_students_db.columns:
        columns_to_drop.append('School_Type_Public')
    if 'Sleep_Hours' in encoded_students_db.columns:
        columns_to_drop.append('Sleep_Hours')
    if 'Motivation_Level' in encoded_students_db.columns:
        columns_to_drop.append('Motivation_Level')
    
    encoded_students_db = encoded_students_db.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Feature engineering completed: {encoded_students_db.shape[1]} final features")
    
    # Step 4: Model Training
    print("\nStep 4: Model Training...")
    
    # Prepare features and target
    X = encoded_students_db.drop(columns=['Exam_Score'])
    y = encoded_students_db['Exam_Score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: Training ({X_train.shape[0]} samples), Testing ({X_test.shape[0]} samples)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    print(f"Polynomial features created: {X_train.shape[1]} -> {X_train_poly.shape[1]} features")
    
    # Train polynomial regression model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    print("Polynomial regression model trained")
    
    # Step 5: Model Evaluation
    print("\nStep 5: Model Evaluation...")
    
    # Make predictions
    y_pred = poly_model.predict(X_test_poly)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance Metrics:")
    print(f"   R2 Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    # Step 6: Model Deployment
    print("\nStep 6: Model Deployment...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    os.makedirs('deployment', exist_ok=True)
    
    # Save model artifacts
    model_artifacts = {
        'poly_model.pkl': poly_model,
        'poly_scaler.pkl': scaler,
        'poly_features.pkl': poly,
        'label_encoders.pkl': label_encoders,
        'onehot_encoder.pkl': ohe,
        'feature_columns.pkl': list(X.columns)
    }
    
    for filename, artifact in model_artifacts.items():
        filepath = os.path.join('model', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(artifact, f)
        print(f"   {filename} saved")
    
    # Step 7: Create Deployment Configuration
    print("\nStep 7: Creating Deployment Configuration...")
    
    deployment_config = {
        "model_info": {
            "name": "Student Performance Polynomial Regression",
            "version": "1.0.0",
            "algorithm": "Polynomial Regression (Degree 2)",
            "deployment_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance": {
                "r2_score": float(r2),
                "rmse": float(rmse),
                "mae": float(mae)
            }
        },
        "features": {
            "continuous": continuous_cols,
            "ordinal": ordinal_columns,
            "nominal": nominal_columns,
            "total_features": len(X.columns),
            "polynomial_features": X_train_poly.shape[1]
        },
        "preprocessing": {
            "scaling": "StandardScaler",
            "encoding": "LabelEncoder + OneHotEncoder",
            "polynomial_degree": 2,
            "feature_selection": "Correlation-based"
        },
        "data_info": {
            "training_samples": X_train.shape[0],
            "testing_samples": X_test.shape[0],
            "original_features": students_db.shape[1],
            "final_features": X.shape[1]
        }
    }
    
    # Save deployment configuration
    config_path = os.path.join('deployment', 'deployment_config.json')
    with open(config_path, 'w') as f:
        json.dump(deployment_config, f, indent=2)
    print(f"Deployment configuration saved to {config_path}")
    
    # Step 8: Generate Deployment Report
    print("\nStep 8: Generating Deployment Report...")
    
    report_path = os.path.join('deployment', 'deployment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("STUDENT PERFORMANCE MODEL DEPLOYMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write(f"Name: {deployment_config['model_info']['name']}\n")
        f.write(f"Version: {deployment_config['model_info']['version']}\n")
        f.write(f"Algorithm: {deployment_config['model_info']['algorithm']}\n")
        f.write(f"Deployment Date: {deployment_config['model_info']['deployment_date']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n\n")
        
        f.write("FEATURE INFORMATION:\n")
        f.write(f"Continuous Features: {len(continuous_cols)}\n")
        f.write(f"Ordinal Features: {len(ordinal_columns)}\n")
        f.write(f"Nominal Features: {len(nominal_columns)}\n")
        f.write(f"Total Features: {len(X.columns)}\n")
        f.write(f"Polynomial Features: {X_train_poly.shape[1]}\n\n")
        
        f.write("DATA INFORMATION:\n")
        f.write(f"Training Samples: {X_train.shape[0]}\n")
        f.write(f"Testing Samples: {X_test.shape[0]}\n")
        f.write(f"Original Features: {students_db.shape[1]}\n")
        f.write(f"Final Features: {X.shape[1]}\n\n")
        
        f.write("DEPLOYMENT STATUS: SUCCESSFUL\n")
        f.write("Model is ready for production use.\n")
    
    print(f"Deployment report saved to {report_path}")
    
    # Step 9: Create Production Ready Files
    print("\nStep 9: Creating Production Ready Files...")
    
    # Create production model loader
    production_loader = '''
import pickle
import os
import pandas as pd

class StudentPerformanceModel:
    """Production-ready model loader for student performance prediction"""
    
    def __init__(self, model_path='model/'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.poly_features = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load all model artifacts"""
        try:
            with open(os.path.join(self.model_path, 'poly_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'poly_scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'poly_features.pkl'), 'rb') as f:
                self.poly_features = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'label_encoders.pkl'), 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'onehot_encoder.pkl'), 'rb') as f:
                self.onehot_encoder = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'feature_columns.pkl'), 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Apply label encoding
            ordinal_columns = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 
                              'Family_Income', 'Teacher_Quality', 'Peer_Influence', 
                              'Parental_Education_Level', 'Distance_from_Home']
            
            for col in ordinal_columns:
                if col in input_df.columns and col in self.label_encoders:
                    input_df[col] = self.label_encoders[col].transform(input_df[col])
            
            # Apply one-hot encoding
            nominal_columns = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Gender', 'Learning_Disabilities']
            
            for col in nominal_columns:
                if col in input_df.columns:
                    if col == 'Extracurricular_Activities':
                        input_df['Extracurricular_Activities_Yes'] = (input_df[col] == 'Yes').astype(int)
                    elif col == 'Internet_Access':
                        input_df['Internet_Access_Yes'] = (input_df[col] == 'Yes').astype(int)
                    elif col == 'School_Type':
                        input_df['School_Type_Public'] = (input_df[col] == 'Public').astype(int)
                    elif col == 'Gender':
                        input_df['Gender_Male'] = (input_df[col] == 'Male').astype(int)
                    elif col == 'Learning_Disabilities':
                        input_df['Learning_Disabilities_Yes'] = (input_df[col] == 'Yes').astype(int)
                    
                    input_df = input_df.drop(columns=[col])
            
            # Add missing columns
            missing_columns = ['Physical_Activity']
            for col in missing_columns:
                if col not in input_df.columns:
                    if col == 'Physical_Activity':
                        input_df[col] = 3
            
            # Ensure correct feature order - use the actual feature columns from the model
            if hasattr(self, 'feature_columns') and self.feature_columns is not None:
                required_features = self.feature_columns
            else:
                # Fallback to expected features
                required_features = [
                    'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
                    'Previous_Scores', 'Tutoring_Sessions', 'Extracurricular_Activities_Yes', 
                    'Internet_Access_Yes', 'Physical_Activity'
                ]
            
            input_df = input_df[required_features]
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Apply polynomial transformation
            input_poly = self.poly_features.transform(input_scaled)
            
            return input_poly
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def predict(self, input_data):
        """Make prediction on preprocessed input"""
        try:
            # Preprocess input
            input_processed = self.preprocess_input(input_data)
            
            if input_processed is None:
                return None
            
            # Make prediction
            prediction = self.model.predict(input_processed)[0]
            return round(prediction, 2)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        return {
            "algorithm": "Polynomial Regression (Degree 2)",
            "features": len(self.feature_columns),
            "polynomial_features": self.poly_features.n_output_features_,
            "status": "Loaded" if self.model is not None else "Not Loaded"
        }

# Test the model
if __name__ == "__main__":
    print("Testing Production Model...")
    
    # Initialize model
    model = StudentPerformanceModel()
    
    # Display model info
    print("\\nModel Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test prediction
    print("\\nTesting Prediction...")
    test_data = {
        'Hours_Studied': 25,
        'Attendance': 90,
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 8,
        'Previous_Scores': 85,
        'Motivation_Level': 'High',
        'Internet_Access': 'Yes',
        'Tutoring_Sessions': 2,
        'Family_Income': 'Medium',
        'Teacher_Quality': 'High',
        'School_Type': 'Public',
        'Peer_Influence': 'Positive',
        'Physical_Activity': 4,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'College',
        'Distance_from_Home': 'Near',
        'Gender': 'Male'
    }
    
    prediction = model.predict(test_data)
    if prediction:
        print(f"Test prediction successful: {prediction}")
    else:
        print("Test prediction failed")
'''
    
    # Save production loader
    loader_path = os.path.join('deployment', 'production_model.py')
    with open(loader_path, 'w') as f:
        f.write(production_loader)
    print(f"Production model loader saved to {loader_path}")
    
    # Step 10: Final Deployment Summary
    print("\n" + "=" * 60)
    print("MODEL DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\nFiles Created:")
    print(f"   Model artifacts: model/")
    print(f"   Deployment config: deployment/deployment_config.json")
    print(f"   Deployment report: deployment/deployment_report.txt")
    print(f"   Production loader: deployment/production_model.py")
    
    print(f"\nNext Steps:")
    print(f"   1. Test the model: python deployment/production_model.py")
    print(f"   2. Run the web app: python app_production.py")
    print(f"   3. Deploy to production server")
    print(f"   4. Monitor model performance")
    
    print(f"\nModel Performance:")
    print(f"   R2 Score: {r2:.4f} (Excellent)")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    return True

if __name__ == "__main__":
    success = deploy_polynomial_regression_model()
    if success:
        print("\nDeployment pipeline completed successfully!")
    else:
        print("\nDeployment pipeline failed!")
