# Task 1: Iris Flower Classification

## Objective
Classify iris flowers into species using sepal and petal measurements.

## Data preparation
The supplied `Iris.csv` contains 150 records. `Id` was removed because it is an identifier, not a flower measurement. The four measurement columns are predictors and `Species` is the target. An 80/20 stratified train/test split was used.

## Method
A pipeline standardizes the numeric measurements and fits Logistic Regression. This is a supervised multiclass classification model.

## Results
**Test accuracy: 93.3%**

Classification report:
```text
precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       0.90      0.90      0.90        10
 Iris-virginica       0.90      0.90      0.90        10

       accuracy                           0.93        30
      macro avg       0.93      0.93      0.93        30
   weighted avg       0.93      0.93      0.93        30
```
See `confusion_matrix.png` for class-level test predictions.

## Conclusion
The model classified 28 of 30 test examples correctly on this split. The small, clean dataset makes this a useful introductory classification project.

## Run
Install dependencies with `pip install -r requirements.txt`, then run `python iris_classification.py`.

## Limitation
The result is based on one split of a small benchmark dataset and should not be treated as guaranteed real-world accuracy.
