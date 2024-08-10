# Diabetes Detection using Naive Bayes

## Overview

This project implements a diabetes detection system using machine learning techniques, specifically Naive Bayes. The goal is to predict whether a patient has diabetes based on various medical attributes.

## Features

- **Data Preprocessing:** Handling missing values, data normalization, and feature engineering.
- 
- **Model Training:** Naive Bayes classifier to predict diabetes presence.
- 
- **Evaluation:** Model performance evaluation with metrics such as accuracy.
- 
- **Visualization:** Insights into the dataset and model performance.

## Dataset

The dataset used in this project contains medical information about patients. You can download the dataset from the following link:

- [Download Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Data Description

The dataset includes the following predictor variables:

- **Pregnancies:** Number of times the patient got pregnant.
  
- **Glucose:** Plasma glucose concentration.
  
- **Blood Pressure:** Diastolic Blood Pressure (mmHg).
  
- **Skin Thickness:** Triceps skin fold thickness (mm).
  
- **Insulin:** 2-Hour serum insulin (mu U/ml).
  
- **BMI:** Body mass index (weight in kg/(height in m)^2).

- **DiabetesPedigreeFunction:** Diabetes pedigree function.

- **Age:** Age (years).

The outcome variable is:

- **Outcome:** Class variable (0 or 1), where 1 indicates the presence of diabetes and 0 indicates the absence.

## Installation

1. **Clone the Repository:**

   git clone https://github.com/srus1608/Diabetes-Detection-Naive-Bayes.git
   cd Diabetes-Detection-Naive-Bayes

2. Install Required Libraries:

Make sure you have Python installed. Then, install the required libraries using pip:

pip install -r requirements.txt
If requirements.txt is not available, install libraries individually:

pip install numpy pandas scikit-learn matplotlib seaborn

## Usage
Load the Dataset:

Ensure that the dataset is in the correct directory or update the path in the code:

import pandas as pd
data = pd.read_csv('path_to_your_dataset.csv')

## Run the Notebook:

Open the Jupyter notebook:

jupyter notebook Diabetes_Detection_Naive_Bayes.ipynb
Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

## Results
The Naive Bayes model is evaluated based on various metrics. Key results include:

Accuracy: 77.92%

Precision: The proportion of true positives out of all positive predictions.

Recall: The proportion of true positives out of all actual positives.

F1 Score: The harmonic mean of precision and recall.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. For detailed guidelines, please refer to the CONTRIBUTING.md file.



