from flask import Flask, render_template, request, jsonify
from production_model import StudentPerformanceModel
import traceback

app = Flask(__name__)

# Initialize production model
print("🚀 Initializing Production Model...")
model = StudentPerformanceModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for production monitoring"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model.model is not None,
        "model_info": model.get_model_info(),
        "service": "Student Performance Predictor API"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint using production model"""
    try:
        # Get input data from the form
        data = {
            'Hours_Studied': int(request.form['hours_studied']),
            'Attendance': int(request.form['attendance']),
            'Parental_Involvement': request.form['parental_involvement'],
            'Access_to_Resources': request.form['access_to_resources'],
            'Extracurricular_Activities': request.form['extracurricular_activities'],
            'Sleep_Hours': int(request.form['sleep_hours']),
            'Previous_Scores': int(request.form['previous_scores']),
            'Motivation_Level': request.form['motivation_level'],
            'Internet_Access': request.form['internet_access'],
            'Tutoring_Sessions': int(request.form['tutoring_sessions']),
            'Family_Income': request.form['family_income'],
            'Teacher_Quality': request.form['teacher_quality'],
            'School_Type': request.form['school_type'],
            'Peer_Influence': request.form['peer_influence'],
            'Physical_Activity': int(request.form['physical_activity']),
            'Learning_Disabilities': request.form['learning_disabilities'],
            'Parental_Education_Level': request.form['parental_education_level'],
            'Distance_from_Home': request.form['distance_from_home'],
            'Gender': request.form['gender']
        }
        
        # Make prediction using production model
        prediction = model.predict(data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed', 'success': False}), 500
        
        # Determine grade based on score
        if prediction >= 90:
            grade = "A+"
        elif prediction >= 85:
            grade = "A"
        elif prediction >= 80:
            grade = "A-"
        elif prediction >= 75:
            grade = "B+"
        elif prediction >= 70:
            grade = "B"
        elif prediction >= 65:
            grade = "B-"
        elif prediction >= 60:
            grade = "C+"
        elif prediction >= 55:
            grade = "C"
        else:
            grade = "C-"
        
        return jsonify({
            'prediction': prediction,
            'grade': grade,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/model-info')
def model_info():
    """Endpoint to get detailed model information"""
    return jsonify({
        'model_info': model.get_model_info(),
        'features': model.feature_columns,
        'status': 'Production Ready'
    })

if __name__ == '__main__':
    if model.model is None:
        print("❌ Model not loaded. Please check model files.")
        exit(1)
    
    print("✅ Production model loaded successfully!")
    print("🌐 Starting Flask application...")
    app.run(debug=False, host='0.0.0.0', port=5000)
