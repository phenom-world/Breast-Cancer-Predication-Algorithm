# Breast Cancer Prediction Algorithm

A machine learning model to predict breast cancer diagnosis based on various medical features. The model achieves over 90% accuracy using Support Vector Machine (SVM) classification.

## Overview

This project uses the UCTH Breast Cancer Dataset to build a predictive model that can help classify breast tumors as either benign or malignant. The model considers various factors including:

- Age
- Menopause status
- Tumor size
- Breast quadrant location
- Patient history

## Models Implemented

Several machine learning models were trained and evaluated:

1. Support Vector Machine (SVM)
   - Best performing model with 90% accuracy
   - Uses linear kernel with C=0.05

2. Random Forest Classifier
   - 86% test accuracy
   - 97% training accuracy

3. Logistic Regression and KNN models were also evaluated

## Getting Started

### Prerequisites


⌘K to generate a command
Write a proper readme and create the frontend to test

I'll help you create a comprehensive README.md and suggest a frontend implementation for testing the breast cancer prediction model.
First, let's update the README:
# Breast Cancer Prediction Algorithm

A machine learning model to predict breast cancer diagnosis based on various medical features. The model achieves over 90% accuracy using Support Vector Machine (SVM) classification.

## Overview

This project uses the UCTH Breast Cancer Dataset to build a predictive model that can help classify breast tumors as either benign or malignant. The model considers various factors including:

- Age
- Menopause status
- Tumor size
- Breast quadrant location
- Patient history

## Models Implemented

Several machine learning models were trained and evaluated:

1. Support Vector Machine (SVM)
   - Best performing model with 90% accuracy
   - Uses linear kernel with C=0.05

2. Random Forest Classifier
   - 86% test accuracy
   - 97% training accuracy

3. Logistic Regression and KNN models were also evaluated

## Getting Started

### Prerequisites

bash
pip install -r requirements.txt

bash
git clone https://github.com/yourusername/Breast-Cancer-Prediction-Algorithm.git
cd Breast-Cancer-Prediction-Algorithm

bash
pip install pandas numpy scikit-learn joblib

python
import joblib
Load the model
model = joblib.load('svc.pkl')
Make predictions
Example input format:
sample = [[40, 1, 2, "Upper inner", 0]] # Age, Menopause, Tumor Size, Breast Quadrant, History
prediction = model.predict(sample)


## Model Performance

- Accuracy: 90%
- High precision in identifying malignant cases
- Good recall rate for minimizing false negatives

## Dataset

The dataset is from the University College Teaching Hospital (UCTH) and contains:
- 213 patient records
- 6 key features
- Binary classification (Benign/Malignant)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.