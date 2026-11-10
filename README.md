# Machine Learning-Based Menstrual Cycle Length Prediction and Explainable AI Analysis

## 1. Project Overview

This project explores the application of Machine Learning (ML) and Explainable Artificial Intelligence (XAI) to analyse menstrual health data and predict menstrual cycle length.

The project uses selected menstrual health features, including age, BMI, menstrual duration, luteal phase length, number of peaks, unusual bleeding, and menses score, to develop regression models for predicting menstrual cycle length.

Four machine learning algorithms are implemented and compared. SHAP (SHapley Additive exPlanations) is used to interpret the predictions and understand the contribution of individual features to the model's output.

The primary goal is to develop an interpretable machine learning workflow for menstrual health data analysis.

## 2. Why This Project?

Menstrual cycle characteristics can vary between individuals and may be influenced by several physiological and lifestyle-related factors.

Machine learning can help explore patterns in menstrual health data and investigate whether selected features can be used to predict cycle length.

However, machine learning models are not always easy to interpret. Therefore, this project incorporates Explainable AI using SHAP to understand how different input features influence model predictions.

This project was developed to:

- Explore menstrual health data using Python.
- Apply machine learning techniques to predict menstrual cycle length.
- Compare the performance of different regression algorithms.
- Evaluate model performance using appropriate statistical metrics.
- Apply SHAP to understand feature contributions.
- Develop practical skills in healthcare data analysis and explainable machine learning.

## 3. Dataset Description

**Dataset:** Menstrual Health and PCOD Risk Detection Dataset

The dataset contains menstrual health-related variables that can be used for exploratory analysis and cycle length prediction.

The original dataset contained 162 records and 13 columns. After removing 81 duplicate records and performing data cleaning, 81 unique records were retained for analysis.

### Selected Features

    number_of_peak 
    Age 
    Length_of_Leutal_Phase 
    Length_of_menses 
    Unusual_Bleeding 
    BMI 
    Menses_score

**Target Variable:** `Length_of_cycle`

The target variable represents the menstrual cycle length that the models aim to predict.

Some variables from the original dataset were excluded because of potential redundancy or the possibility of target leakage.

**Note:** The dataset does not provide a verified PCOD diagnosis label. Therefore, this project focuses on menstrual cycle length prediction rather than PCOD diagnosis or classification.

## 4. Machine Learning Algorithms

Four regression algorithms were implemented to predict menstrual cycle length:

    1.Linear Regression -------> Estimates the relationship between input features and the target using a linear equation. 
    2.Random Forest Regressor -------> Combines multiple decision trees to generate predictions. 
    3.Gradient Boosting Regressor -------> Builds sequential decision trees to improve prediction performance. 
    4.Support Vector Regression (SVR) -------> Uses support vector methods to model relationships between input features and continuous target values. 

### Data Preprocessing

The following preprocessing techniques were applied:

- Removal of duplicate records.
- Selection of relevant input features.
- Separation of independent variables and target variable.
- Train-test splitting.
- One-Hot Encoding for categorical features.
- StandardScaler for numerical features.

Preprocessing transformations were fitted using training data only to reduce data leakage.

## 5. Model Evaluation

The regression models were evaluated using the following performance metrics:

    Mean Absolute Error (MAE) -------> Measures the average absolute difference between actual and predicted values. 
    Mean Squared Error (MSE) -------> Measures the average squared prediction error. 
    Root Mean Squared Error (RMSE) -------> Measures prediction error in the same unit as the target variable. 
    R² Score -------> Measures how much variation in the target is explained by the model. 

### Cross-Validation

Five-fold cross-validation was performed using KFold to evaluate model performance across different subsets of the dataset.

A baseline model using the mean prediction strategy was also implemented for comparison.

## 6. Explainable AI Using SHAP

SHAP (SHapley Additive exPlanations) was implemented to understand the prediction behaviour of the Support Vector Regression model.

SHAP estimates the contribution of individual input features to model predictions.

The following visualisations were generated:

- **SHAP Summary Plot:** Shows how different feature values contribute positively or negatively to model predictions.
- **SHAP Feature Importance Plot:** Ranks features based on their mean absolute SHAP values.

### Observations

Based on the SHAP analysis:

1. `number_of_peak` showed the highest average feature contribution.
2. `Length_of_Leutal_Phase` was the second most influential feature.
3. BMI and Menses_score also contributed to the model predictions.
4. Age, Length_of_menses, and Unusual_Bleeding showed comparatively lower average contributions.

These observations represent the behaviour of the trained model and should not be interpreted as biological cause-and-effect relationships.

## 7. Results and Findings

### Cross-Validation Results

The following results were obtained using five-fold cross-validation.

| Model                          | Mean MAE          | Mean RMSE          | Mean R² |

| Linear Regression              | 3.151             | 4.066              | -0.006 |
| Random Forest                  | 3.532             | 4.368              | -0.211 |
| Gradient Boosting              | 3.535             | 4.466              | -0.262 |
| Support Vector Regression      | 2.987             | 4.090              | -0.012 |

### Key Findings

- Support Vector Regression achieved the lowest mean MAE (2.987) and mean RMSE (4.090) among the four machine learning models.
- Linear Regression achieved the R² value closest to zero (-0.006).
- The R² values indicate limited predictive performance across the evaluated models.
- The baseline model achieved a mean MAE of approximately 3.071, which was slightly higher than the SVR mean MAE.
- SHAP analysis identified `number_of_peak` as the feature with the highest average contribution to the SVR predictions.

Overall, SVR performed comparatively better in terms of average prediction errors, but the results do not demonstrate strong predictive ability.

## 8. Limitations

The project has several limitations:

1. **Small Dataset:** The cleaned dataset contains only 81 unique records, which limits the ability of machine learning models to learn generalisable patterns.

2. **Limited Predictive Performance:** The evaluated models produced R² values close to zero or negative, indicating weak predictive performance.

3. **Feature Availability:** The analysis is restricted to the variables available in the dataset. Additional relevant physiological and lifestyle-related features may improve future investigations.

4. **Model Generalisation:** The models have not been externally validated using an independent dataset.

5. **SHAP Interpretation:** SHAP explains the behaviour of the trained model but does not establish biological causation. The current explanations were generated using the same dataset used to train the model, so their stability requires further investigation.

6. **No Clinical Validation:** The predictions are experimental and should not be used for medical diagnosis or clinical decision-making.

## 9. Future Improvements

- Increase the dataset size using reliable and well-documented data sources.
- Validate the definitions and quality of the selected features.
- Evaluate additional regression algorithms and hyperparameter optimisation techniques.
- Perform external validation using independent datasets.
- Investigate the stability of SHAP feature importance across different data splits.
- Explore additional explainability methods to improve model interpretation.

## 10. Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Explainable AI:** SHAP
- **Development Environment:** Jupyter Notebook

## 11.Installation 

Follow the steps below to set up and run this project on your local system.

### 1. Clone the Repository

```bash
git clone https://github.com/rithinika29/Menstrual_Health_ML_Project.git
```

### 2. Navigate to the Project Directory

```bash
cd Menstrual_Health_ML_Project
```

### 3. Install Required Libraries

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebooks

Launch Jupyter Notebook using:

```bash
jupyter notebook
```

Open the `Notebooks` folder and run the notebooks in numerical order, starting from `01_Data_Exploration.ipynb`.



## 12. Project Structure

```text
Menstrual_Health_ML_Project/
│
├── Dataset/
│   └── menstrual_health_cleaned.csv
│
├── Notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Feature_Selection.ipynb
│   ├── 03_Data_Preprocessing.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   ├── 06_Cross_Validation.ipynb
│   ├── 07_SHAP_Explainability.ipynb
│   ├── 08_Final_Visualisation_and_Reporting.ipynb
│   └── 09_New_Cycle_Length_Prediction.ipynb
│
├── Visualizations/
│
├── README.md
│
└── requirements.txt
```

## 13. Disclaimer

This project is developed for educational and research purposes. It demonstrates the application of machine learning and explainable AI techniques to menstrual health data. The predictions are not clinically validated and should not be considered medical advice or a substitute for professional healthcare assessment.