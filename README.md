## BLENDED_LEARNING
## Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients
### DATE:24-04-2025
## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Libraries**  
   Import necessary Python libraries for data manipulation, model building, and evaluation.

2. **Load Dataset**  
   Load the nutritional information dataset using `pandas`.

3. **Explore Dataset**  
   Display the first few rows and data types to understand the structure and contents.

4. **Preprocess Data**  
   - Separate features (`X_raw`) and target (`y_raw`).  
   - Scale the features using `MinMaxScaler`.  
   - Encode the target labels using `LabelEncoder`.

5. **Split Dataset**  
   Split the dataset into training and testing sets using `train_test_split`.

6. **Build Logistic Regression Model**  
   Define and train the logistic regression model with parameters suitable for multiclass classification.

7. **Predict on Test Data**  
   Use the trained model to predict labels for the test set.

8. **Evaluate Model**  
   Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

9. **Visualize Confusion Matrix**  
   Plot the confusion matrix using `seaborn` to visually assess classification performance.


## Program:
```
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: SREE HARI K
RegisterNumber: 212223230212

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('food_items.csv')

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]
X_raw
scaler = MinMaxScaler()
# Scaling the raw input features.
X = scaler.fit_transform(X_raw)

# Create a LabelEncoder object
label_encoder = LabelEncoder()
# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())
# Note that ravel() function flattens the vector.

# First, let's split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
# L2 penalty to shrink coefficients without removing any features from the model
penalty = 'l2'

# Our classification problem is multinomial
multi_class = 'multinomial'

# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'

# Max iteration = 1000
max_iter = 1000

# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

l2_model.fit(X_train, y_train)
y_pred = l2_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

```

## Output:
### DATA OVERVIEW:
![image](https://github.com/user-attachments/assets/62545e8f-c647-4068-b6f4-8e5591909c88)
### DATA INFO:
![image](https://github.com/user-attachments/assets/509a8b58-c904-4988-92bd-d6a3f66c3bee)
### PREVIEW OF FOOD NUTRITION DATA:
![image](https://github.com/user-attachments/assets/32b14d04-308c-4e12-9bf0-8b22b7d73e5c)
### EVALUATION OF THE MODEL:
![image](https://github.com/user-attachments/assets/25af3fdf-eb0f-4735-a623-e8183f0e01bb)
### CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/38aab9de-f8c9-4f1f-aca8-6b98b462a337)



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
