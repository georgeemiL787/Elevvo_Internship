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
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
            print(f"❌ Error in preprocessing: {e}")
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
            print(f"❌ Error in prediction: {e}")
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
    print("🧪 Testing Production Model...")
    
    # Initialize model
    model = StudentPerformanceModel()
    
    # Display model info
    print("\n📊 Model Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test prediction
    print("\n🧪 Testing Prediction...")
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
        print(f"✅ Test prediction successful: {prediction}")
    else:
        print("❌ Test prediction failed")
