import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

st.title("Fine Needle Aspiration - Prediction using Logistic Regression")
dataset = sklearn.datasets.load_breast_cancer()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

df['label'] = dataset.target

df['label'].value_counts()

df.groupby('label').mean()

X = df.drop(columns='label', axis=1)
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

trainingModel = LogisticRegression()


trainingModel.fit(X_train, Y_train)

X_train_prediction = trainingModel.predict(X_train)
accuracy = accuracy_score(Y_train, X_train_prediction)
st.text("Training Accuracy :")
st.caption(accuracy)

X_test_prediction = trainingModel.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
st.text("Testing Accuracy :")
st.caption(accuracy)
print('Accuracy on test data = ', test_data_accuracy)
st.caption("Placeholder Values Given")
input_data = [];
placeholder = [14.42, 19.77, 94.48, 642.5, 0.09752, 0.1141, 0.09388, 0.05839, 0.1879, 0.0639, 0.2895, 1.851, 2.376,
               26.85, 0.008005, 0.02895, 0.03321, 0.01424, 0.01462, 0.004452, 16.33, 30.86, 109.5, 826.4, 0.1431,
               0.3026, 0.3194, 0.1565, 0.2718, 0.09353]
i = 0;
parameters = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
              "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se",
              "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
              "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
              "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
              "concave points_worst", "symmetry_worst", "fractal_dimension_worst"];
for parameter in parameters:
    input_data.append(float(st.text_input(parameter, value=placeholder[i])))
    i = i + 1;
data = np.asarray(input_data)
input_data_reshaped = data.reshape(1, -1)

prediction = trainingModel.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    st.success('Malignant')

else:
    st.success('Benign')
