import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ["Number of times pregnant", "Plasma glucose concentration", "Diastolic blood pressure",
                    "Triceps skin fold thickness", "2-Hour serum insulin", "Body mass index",
                    "Diabetes pedigree function", "Age", "Outcome"]
    data = pd.read_csv(url, names=column_names)
    return data

data = load_data()

# Splitting the data into training and testing sets
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit interface
st.title('Diabetes Prediction Model')
st.write(f"Accuracy of the model is {accuracy:.2f}")

# User input for making a prediction
st.write("## Enter the patient details to predict diabetes")

pregnancies = st.number_input("Number of times pregnant", min_value=0, value=0)
glucose = st.number_input("Plasma glucose concentration", min_value=0, value=120)
bp = st.number_input("Diastolic blood pressure", min_value=0, value=70)
skin_thickness = st.number_input("Triceps skin fold thickness", min_value=0, value=20)
insulin = st.number_input("2-Hour serum insulin", min_value=0, value=85)
bmi = st.number_input("Body mass index", min_value=0.0, value=32.0)
dpf = st.number_input("Diabetes pedigree function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=21, value=33)

if st.button("Predict Diabetes"):
    user_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(user_data)
    prediction_prob = model.predict_proba(user_data)
    if prediction[0] == 1:
        st.subheader("The model predicts the patient has diabetes with a probability of {:.2f}%.".format(prediction_prob[0][1]*100))
    else:
        st.subheader("The model predicts the patient does not have diabetes with a probability of {:.2f}%.".format(prediction_prob[0][0]*100))
