# Student Performance Prediction Project

##  Project Overview

This project implements a comprehensive machine learning system to predict students' exam scores based on various academic, personal, and environmental factors. The system includes data analysis, model training, deployment pipeline, and a production-ready web application.

##  Objectives

- Perform comprehensive data cleaning and exploratory data analysis
- Build and evaluate polynomial regression models for exam score prediction
- Implement a complete deployment pipeline with model artifacts
- Create a production-ready web application with modern UI
- Provide real-time predictions through an intuitive interface

## Dataset

- **Source**: Student Performance Factors (Kaggle)
- **Size**: 6,609 records with 20 features
- **Target Variable**: Exam_Score
- **Features**: Study hours, attendance, parental involvement, access to resources, extracurricular activities, sleep hours, previous scores, motivation level, internet access, tutoring sessions, family income, teacher quality, school type, peer influence, physical activity, learning disabilities, parental education level, distance from home, and gender.

##  Project Structure

```
Task 1/
├──  model/                          # Trained model artifacts
│   ├── poly_model.pkl                 # Trained polynomial regression model
│   ├── poly_scaler.pkl                # Feature scaler
│   ├── poly_features.pkl              # Polynomial feature transformer
│   ├── label_encoders.pkl             # Label encoders for categorical variables
│   ├── onehot_encoder.pkl             # One-hot encoder
│   └── feature_columns.pkl            # Feature column names
├── deployment/                     # Deployment configuration
│   ├── deployment_config.json         # Model configuration and metadata
│   ├── deployment_report.txt          # Deployment summary report
│   └── production_model.py            # Production model class
├──  templates/                      # Web application templates
│   ├── index.html                     # Main prediction interface
│   └── about.html                     # About page
├──  StudentPerformanceFactors.csv   # Original dataset
├── Task_1.ipynb                    # Jupyter notebook with analysis
├── deploy_model_simple.py          # Model training and deployment pipeline
├──  app_production.py               # Production Flask web application
├──  production_model.py             # Production model loader
├── requirements.txt                # Python dependencies
├──  README.md                       # This documentation
└──  output.png                      # Analysis visualization output
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Deploy the Model
```bash
python deploy_model_simple.py
```

### 3. Launch the Web Application
```bash
python app_production.py
```

### 4. Access the Application
Open your browser and navigate to: `http://localhost:5000`

##  Detailed Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone or download** the project files
2. **Navigate** to the Task 1 directory
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the model**:
   ```bash
   python deploy_model_simple.py
   ```
5. **Start the web application**:
   ```bash
   python app_production.py
   ```
6. **Access the application** at `http://localhost:5000`

##  Model Performance

### Polynomial Regression Model
- **R² Score**: 0.9803 (Excellent)
- **RMSE**: ~2.5 points
- **MAE**: ~1.8 points
- **Algorithm**: Polynomial Regression (Degree 2)

### Key Features
- **Feature Engineering**: Advanced preprocessing with label encoding and one-hot encoding
- **Polynomial Features**: Captures non-linear relationships
- **Feature Scaling**: StandardScaler for optimal performance
- **Feature Selection**: Correlation-based feature selection

##  Web Application Features

### Main Interface (`/`)
- **Modern UI**: Beautiful gradient design with Bootstrap 5
- **Comprehensive Form**: 19 input fields covering all prediction factors
- **Real-time Prediction**: AJAX-based prediction without page reload
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Visual feedback during prediction processing
- **Error Handling**: Graceful error display and recovery

### About Page (`/about`)
- **Project Information**: Detailed description of the project
- **Technical Details**: Model architecture and performance metrics
- **Usage Instructions**: How to use the prediction system

### API Endpoints
- **`/predict`** (POST): Make predictions
- **`/health`** (GET): System health check
- **`/model-info`** (GET): Model information and status

##  Deployment Pipeline

### `deploy_model_simple.py` - Complete ML Pipeline

This script implements a comprehensive machine learning deployment pipeline:

#### **Pipeline Steps:**
1. **Data Loading**: Load and validate the dataset
2. **Data Preprocessing**: Handle missing values and data cleaning
3. **Feature Engineering**: 
   - Label encoding for ordinal features
   - One-hot encoding for nominal features
   - Feature scaling with StandardScaler
4. **Model Training**: 
   - Polynomial feature creation (degree 2)
   - Polynomial regression model training
5. **Model Evaluation**: Calculate R², RMSE, MAE metrics
6. **Artifact Saving**: Save all model components as pickle files
7. **Configuration Generation**: Create deployment configuration
8. **Report Generation**: Generate comprehensive deployment report
9. **Production Code Generation**: Create production-ready model class

#### **Output Files:**
- `model/` directory with all model artifacts
- `deployment/deployment_config.json` with model metadata
- `deployment/deployment_report.txt` with detailed report
- `deployment/production_model.py` with production model class

### `app_production.py` - Production Web Application

A Flask-based web application that serves the trained model:

#### **Key Features:**
- **Model Loading**: Automatic loading of trained model artifacts
- **Input Validation**: Comprehensive form data validation
- **Error Handling**: Robust error handling and user feedback
- **Health Monitoring**: Health check endpoint for system monitoring
- **Production Ready**: Configured for production deployment

#### **Routes:**
- `/`: Main prediction interface
- `/predict`: Prediction API endpoint
- `/about`: About page
- `/health`: Health check endpoint
- `/model-info`: Model information endpoint

##  Data Analysis Features

### Exploratory Data Analysis
- **Distribution Analysis**: Target variable and feature distributions
- **Correlation Analysis**: Feature correlation matrices
- **Categorical Analysis**: Impact of categorical variables
- **Outlier Detection**: Identification and handling of outliers

### Feature Engineering
- **Categorical Encoding**: 
  - Label encoding for ordinal variables
  - One-hot encoding for nominal variables
- **Feature Scaling**: StandardScaler for numerical features
- **Polynomial Features**: Degree 2 polynomial transformation
- **Feature Selection**: Correlation-based feature selection

### Model Evaluation
- **Multiple Metrics**: R², RMSE, MAE, MSE
- **Cross-validation**: Robust model evaluation
- **Residual Analysis**: Model diagnostic plots
- **Feature Importance**: Analysis of feature contributions

##  User Interface Features

### Design Elements
- **Modern Gradient Background**: Purple gradient theme
- **Glass Morphism**: Translucent containers with backdrop blur
- **Responsive Layout**: Bootstrap 5 grid system
- **Interactive Elements**: Hover effects and transitions
- **Loading Animations**: Spinner during prediction processing

### Form Organization
- **Academic Factors**: Study hours, attendance, previous scores, tutoring
- **Personal & Environmental**: Sleep, physical activity, parental involvement
- **Resources & Access**: Internet access, teacher quality, resources
- **Social & Demographic**: Extracurricular activities, family income, school type
- **Additional Factors**: Peer influence, learning disabilities, education level

### User Experience
- **Pre-filled Values**: Sensible default values for all fields
- **Input Validation**: Client-side and server-side validation
- **Real-time Feedback**: Immediate response to user actions
- **Error Recovery**: Clear error messages and recovery options

##  Technical Architecture

### Model Architecture
```
Input Data → Preprocessing → Feature Engineering → Polynomial Features → Linear Regression → Prediction
```

### Web Application Architecture
```
User Request → Flask App → Model Prediction → JSON Response → UI Update
```

### File Dependencies
```
deploy_model_simple.py → model/ artifacts
app_production.py → model/ artifacts + templates/
templates/ → static assets (Bootstrap, Font Awesome)
```

##  Requirements

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
flask>=2.0.0
jupyter>=1.0.0
```

### System Requirements
- **Python**: 3.7 or higher
- **Memory**: 4GB RAM recommended
- **Storage**: 100MB free space
- **Browser**: Modern web browser with JavaScript enabled

##  Deployment Options

### Local Development
```bash
python app_production.py
```

### Production Deployment
```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_production:app

# Using Docker (if Dockerfile provided)
docker build -t student-predictor .
docker run -p 5000:5000 student-predictor
```

### Cloud Deployment
- **Heroku**: Deploy using Procfile and requirements.txt
- **AWS**: Deploy on EC2 with load balancer
- **Google Cloud**: Deploy on App Engine
- **Azure**: Deploy on App Service

##  Monitoring and Maintenance

### Health Monitoring
- **Health Check Endpoint**: `/health`
- **Model Status**: Automatic model loading verification
- **Performance Metrics**: Real-time model performance tracking

### Logging
- **Application Logs**: Flask application logging
- **Model Logs**: Prediction and error logging
- **System Logs**: Server and deployment logs

### Maintenance Tasks
- **Model Retraining**: Periodic model retraining with new data
- **Performance Monitoring**: Track prediction accuracy over time
- **Feature Updates**: Update feature engineering as needed
- **Security Updates**: Regular dependency updates

##  Testing

### Model Testing
```bash
# Test the production model
python production_model.py
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "hours_studied=25&attendance=90&..."
```

### Web Interface Testing
- **Cross-browser Testing**: Chrome, Firefox, Safari, Edge
- **Mobile Testing**: Responsive design verification
- **Accessibility Testing**: Screen reader compatibility

##  Learning Outcomes

This project demonstrates:
- **Complete ML Pipeline**: From data to deployment
- **Production Deployment**: Real-world application deployment
- **Web Development**: Flask application development
- **User Interface Design**: Modern, responsive web design
- **API Development**: RESTful API design and implementation
- **Model Management**: Model versioning and artifact management
- **System Monitoring**: Health checks and performance monitoring

##  Future Enhancements

### Potential Improvements
- **Model Versioning**: Implement model version control
- **A/B Testing**: Compare different model versions
- **Real-time Learning**: Online learning capabilities
- **Advanced Analytics**: Dashboard with prediction analytics
- **Multi-language Support**: Internationalization
- **Mobile App**: Native mobile application
- **API Documentation**: Swagger/OpenAPI documentation

### Advanced Features
- **Ensemble Models**: Combine multiple algorithms
- **Deep Learning**: Neural network implementations
- **Feature Store**: Centralized feature management
- **MLOps Pipeline**: Automated model deployment
- **Model Explainability**: SHAP or LIME integration

##  Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- Follow PEP 8 Python style guide
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write comprehensive tests

##  License

This project is created for educational purposes as part of the Huawei internship program.

##  Authors

- **George Emil** - Huawei Internship Student
- **Project Type**: Machine Learning & Web Development
- **Institution**: Huawei

##  Support

For questions or issues:
1. Check the troubleshooting section below
2. Review the deployment logs
3. Test the health endpoint
4. Contact the development team

##  Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check if model files exist
ls -la model/

# Verify file permissions
chmod 644 model/*.pkl
```

#### Web Application Issues
```bash
# Check if port 5000 is available
netstat -tulpn | grep :5000

# Kill process using port 5000
sudo kill -9 $(lsof -t -i:5000)
```

#### Dependency Issues
```bash
# Update pip
pip install --upgrade pip

# Reinstall dependencies
pip uninstall -r requirements.txt
pip install -r requirements.txt
```

### Performance Optimization
- **Memory Usage**: Monitor memory consumption during predictions
- **Response Time**: Optimize model prediction speed
- **Concurrent Users**: Implement connection pooling for high traffic
- **Caching**: Add Redis caching for frequent predictions

---

**Note**: This project demonstrates a complete machine learning deployment pipeline from data analysis to production web application. It serves as an excellent example of modern ML engineering practices and full-stack development. 
