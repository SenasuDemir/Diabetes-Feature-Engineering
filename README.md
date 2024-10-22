# Diabetes Prediction using Random Forest Classifier

This project involves building a machine learning model to predict whether an individual has diabetes based on various health metrics. The model is trained on the **Pima Indian Diabetes dataset** provided by the National Institute of Diabetes and Digestive and Kidney Diseases.

## Dataset

The dataset used in this project is available on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download).


## Project Overview

- **Dataset**: Pima Indian Diabetes dataset
- **Model**: Random Forest Classifier
- **Target Variable**: `Outcome` (1 = Diabetic, 0 = Non-diabetic)
- **Features**:
  - `Pregnancies`: Number of pregnancies
  - `Glucose`: Glucose level
  - `BloodPressure`: Diastolic blood pressure (mm Hg)
  - `SkinThickness`: Skin thickness (mm)
  - `Insulin`: Insulin level (mu U/ml)
  - `BMI`: Body Mass Index (weight in kg/(height in m)^2)
  - `DiabetesPedigreeFunction`: Family history likelihood of diabetes
  - `Outcome`: Diabetes status (1 for diabetic, 0 for non-diabetic)

## Feature Engineering

To improve model performance, the following steps were implemented:

- Handling missing values, especially in the `Insulin` and `SkinThickness` columns.
- Scaling the numeric features to ensure uniformity.
- Creating interaction features between certain variables, such as `Glucose` and `BMI`.

## Model: Random Forest Classifier

The model was built using the **RandomForestClassifier** from the scikit-learn library. Random Forest was selected for its capability to manage both linear and non-linear relationships and provide a balance between interpretability and performance.

### Performance Metrics

The model's performance was measured using multiple evaluation metrics:

| Metric        | **Value (Engineered)** | **Value (Baseline)** |
|---------------|------------------------|----------------------|
| **Accuracy**  | 0.79                   | 0.77                 |
| **Recall**    | 0.72                   | 0.706                |
| **Precision** | 0.67                   | 0.59                 |
| **F1 Score**  | 0.69                   | 0.64                 |
| **AUC**       | 0.77                   | 0.75                 |


