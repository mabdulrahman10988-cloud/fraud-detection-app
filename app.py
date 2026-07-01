import streamlit as st
import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details, or load an example, to check if it's fraud.")

feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

FRAUD_EXAMPLE = [406.0, -2.312226542, 1.951992011, -1.609850732, 3.997905588, -0.522187865, -1.426545319, -2.537387306, 1.391657248, -2.770089277, -2.772272145, 3.202033207, -2.899907388, -0.595221881, -4.289253782, 0.38972412, -1.14074718, -2.830055675, -0.016822468, 0.416955705, 0.126910559, 0.517232371, -0.035049369, -0.465211076, 0.320198199, 0.044519167, 0.177839798, 0.261145003, -0.143275875, 0.0]

NORMAL_EXAMPLE = [0.0, -1.359807134, -0.072781173, 2.536346738, 1.378155224, -0.33832077, 0.462387778, 0.239598554, 0.098697901, 0.36378697, 0.090794172, -0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.468176972, -0.470400525, 0.207971242, 0.02579058, 0.40399296, 0.251412098, -0.018306778, 0.277837576, -0.11047391, 0.066928075, 0.128539358, -0.189114844, 0.133558377, -0.021053053, 149.62]

col1, col2 = st.columns(2)
if col1.button("Load Fraud Example"):
    st.session_state.vals = FRAUD_EXAMPLE
if col2.button("Load Normal Example"):
    st.session_state.vals = NORMAL_EXAMPLE

if "vals" not in st.session_state:
    st.session_state.vals = [0.0] * 30

inputs = []
for i, name in enumerate(feature_names):
    value = st.number_input(name, value=float(st.session_state.vals[i]))
    inputs.append(value)

if st.button("Check Transaction"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0]
    fraud_confidence = proba[1] * 100

    if prediction == 1:
        st.error(f"⚠️ FRAUD detected! ({fraud_confidence:.1f}% confidence)")
    else:
        st.success(f"✅ Normal transaction ({100 - fraud_confidence:.1f}% confidence)")
