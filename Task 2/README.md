# Mall Customer Segmentation Analysis

## Project Overview
This project performs customer segmentation analysis on mall customer data using unsupervised machine learning techniques. The goal is to identify distinct customer groups based on their spending behavior and income patterns to enable targeted marketing strategies.

## Dataset Description
The analysis uses the Mall Customers dataset containing information about 200 customers:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Customer gender (Male/Female)
- **Age**: Customer age in years
- **Annual Income (k$)**: Annual income in thousands of dollars
- **Spending Score (1-100)**: Spending score assigned by the mall based on customer behavior

## Analysis Objectives
1. **Data Exploration**: Understand the distribution and characteristics of customer data
2. **Feature Analysis**: Examine relationships between different customer attributes
3. **Customer Segmentation**: Apply clustering algorithms to identify customer groups
4. **Cluster Evaluation**: Assess the quality of segmentation using performance metrics
5. **Business Insights**: Extract actionable insights for marketing strategies

## Methodology
The analysis follows these key steps:

### 1. Data Preprocessing
- Load and explore the dataset structure
- Check for missing values and data quality
- Perform descriptive statistics analysis

### 2. Exploratory Data Analysis
- Visualize distributions of key variables (Age, Income, Spending Score)
- Analyze gender distribution
- Create correlation heatmaps to understand feature relationships
- Generate pairplots for numerical feature interactions

### 3. Feature Engineering
- Scale numerical features using StandardScaler for normalization
- Focus on Annual Income and Spending Score for clustering (most relevant features)
- Split data into training (80%) and test (20%) sets with random_state=42
- Remove CustomerID column as it's not relevant for clustering
- Handle categorical variables (Gender) appropriately for analysis

### 4. Clustering Analysis
- **K-Means Clustering**: Applied with optimal k=5 clusters and random_state=42
- **DBSCAN Clustering**: Density-based clustering with eps=0.5 and min_samples=5
- Both algorithms are evaluated using silhouette scores for cluster quality assessment
- Visual comparison of clustering results with scatter plots and centroids
- Analysis of cluster characteristics and customer profiles

### 5. Model Evaluation
- **Silhouette Score Analysis**: Quantitative assessment of clustering quality
  - K-Means: Train (0.551), Test (0.567) - Good cluster separation
  - DBSCAN: Train (0.405), Test (0.708) - Mixed performance, better on test set
- **Visual Comparison**: Scatter plots with cluster assignments and centroids
- **Cluster Analysis**: Detailed examination of customer characteristics per cluster
- **Performance Metrics**: Comparison of both algorithms' effectiveness

## Key Findings
- **Dataset Characteristics**: 200 customers with no missing values or duplicates
- **Data Quality**: Clean dataset with 4 numerical and 1 categorical feature
- **Five distinct customer segments** identified through K-Means clustering
- **Significant variation in spending scores** across clusters (17-82 range)
- **DBSCAN provides alternative clustering perspective** with different cluster formations
- **K-Means Silhouette Scores**: Train (0.551), Test (0.567) - indicating good cluster separation
- **DBSCAN Silhouette Scores**: Train (0.405), Test (0.708) - showing mixed performance
- **Feature Scaling**: StandardScaler applied to normalize Annual Income and Spending Score features
- **Data Split**: 80% training, 20% test split with random_state=42 for reproducibility

## Business Applications
The segmentation results can be used for:
- Targeted marketing campaigns
- Personalized customer experiences
- Inventory planning based on customer preferences
- Pricing strategy optimization
- Customer retention programs

## Technical Requirements
- **Python 3.x**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (KMeans, DBSCAN, StandardScaler, train_test_split)
- **Metrics**: silhouette_score for clustering evaluation
- **Environment**: Jupyter notebook environment

## Files Description
- `task_2.ipynb`: Main Jupyter notebook containing the complete analysis
- `Mall_Customers.xls`: Dataset file containing customer information
- `README.md`: This documentation file

## How to Run
1. Ensure all required Python packages are installed
2. Open the Jupyter notebook `task_2.ipynb`
3. Run all cells sequentially to perform the complete analysis
4. The notebook will generate visualizations and clustering results

## Results Summary
The analysis successfully identified 5 customer segments with distinct characteristics:

### K-Means Clustering Results
- **Cluster 0**: 81 customers with moderate spending (avg score: 49.5) - Largest segment
- **Cluster 1**: 23 customers with low spending (avg score: 20.9) - Small segment
- **Cluster 2**: 22 customers with high spending (avg score: 79.4) - High-value segment
- **Cluster 3**: 39 customers with very high spending (avg score: 82.1) - Premium segment
- **Cluster 4**: 35 customers with very low spending (avg score: 17.1) - Budget segment

### Performance Metrics
- **K-Means Silhouette Score**: 0.551 (train), 0.567 (test) - Good cluster quality
- **DBSCAN Silhouette Score**: 0.405 (train), 0.708 (test) - Better test performance
- **Data Distribution**: Balanced representation across different spending categories

### Key Insights
- **Spending Range**: Wide variation from 17.1 to 82.1 average spending scores
- **Segment Sizes**: Uneven distribution with Cluster 0 being the largest (40.5%)
- **Business Value**: Clusters 2 and 3 represent high-value customers for targeted marketing
- **Cluster Stability**: Consistent performance across training and test sets

These segments provide valuable insights for developing targeted marketing strategies and improving customer experience.
