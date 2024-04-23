import numpy as np
import pickle
import streamlit as st

with open('telehealth.pkl', 'rb') as file:
        data = pickle.load(file)

loaded_model = data['model']
le_car_truck = data["le_car_truck"]
le_insurance = data["le_insurance"]
le_education = data["le_education"]
le_occupation = data["le_occupation"]
le_Airtime = data["le_Airtime"]

def income_prediction(input_data):
        # Prepare your input data
        X = np.array([input_data])
        X[:,1]= le_car_truck.transform(X[:,1])
        X[:,2]= le_insurance.transform(X[:,2])
        X[:,3]= le_education.transform(X[:,3])
        X[:,4]= le_occupation.transform(X[:,4])
        X[:,5]= le_Airtime.transform(X[:,5])
        X=X.astype(float)
        
        # Make predictions using the loaded model
        prediction = loaded_model.predict(X)
        
        # Check the prediction and return the result
        if prediction[0] == '<50k':
            return '<div style="color:green;">Earns less than 50k thus, eligible</div>'
        else:
            return '<div style="color:red;">Earns more than 50k thus, ineligible</div>'

def main():
    
    st.title("Income Prediction App")
    st.write("""### Provide Beneficiary Information""")
    
    car_truck = ('No', 'Yes')
    h_insurance = ('No', 'Yes')
    education = ('Higher', 'Secondary', 'Primary or less')
    occupation = ('Sales and services', 'Professional/technical/managerial', 'Others')
    airtime_spend = ('< N100','> N1000', 'N100 - N500', 'N500 - N1000')
    
    Children_number = st.slider("How many children do you have", 0, 20, 1)
    Car_Truck = st.selectbox("Do you own a truck or car", car_truck)
    H_Insurance = st.selectbox("Do you have health insurance", h_insurance)
    Education = st.selectbox("What is your level of education", education)
    Occupation = st.selectbox("What do you do for a living", occupation)
    Airtime_spend = st.selectbox("How much do you spend on airtime per week", airtime_spend)
    
    prediction = ""

    if st.button("Predict Income"):
        prediction = income_prediction([Children_number, Car_Truck, H_Insurance, Education, Occupation, Airtime_spend])
        st.markdown(prediction, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
