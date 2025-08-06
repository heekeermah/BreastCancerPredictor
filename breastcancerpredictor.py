import streamlit as st
import numpy as np
import pickle
import sklearn 

with open('breast_cancer.pkl','rb') as file:
    model = pickle.load(file)


st.title('Breast Cancer Predictor App🔬')
st.markdown("""
### **🩺 Discover whether your tumor is malignant or beningn!**
This app helps predicts what type of breast cancer you have. fill in your details and click **Diagnose** to see your result
""")


radius_mean = st.number_input('Radius Mean', min_value=6, max_value=28, value=6, step=1)
texture_mean = st.number_input('Texture Mean', min_value=9, max_value=40, value=9, step=1)
perimeter_mean = st.number_input('Perimeter Mean', min_value=40, max_value=190, value=40, step=1)
area_mean = st.number_input('Area Mean', min_value=140, max_value=2500, value=140, step=1)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.05, max_value=0.16, value=0.05, step=0.01)
compactness_mean = st.number_input('Compactness Mean', min_value=0.02, max_value=0.35, value=0.02, step=0.01)
concavity_mean = st.number_input('Concavity Mean', min_value=0.00, max_value=0.43, value=0.00, step=0.01)
concave_points_mean = st.number_input('Concave Points Mean', min_value=0.00, max_value=0.20, value=0.00, step=0.01)
symmetry_mean = st.number_input('Symmetry Mean', min_value=0.12, max_value=0.30, value=0.12, step=0.01)
fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.05, max_value=0.10, value=0.05, step=0.01)
radius_se = st.number_input('Radius Standard Error', min_value=0.10, max_value=0.29, value=0.10, step=0.01)
texture_se = st.number_input('Texture Standard Error', min_value=0.40, max_value=5.00, value=0.40, step=0.01)
perimeter_se = st.number_input('Perimeter Standard Error', min_value=0.75, max_value=22.00, value=0.75, step=0.01)
area_se = st.number_input('Area Standard Error', min_value=6, max_value=550, value=6, step=1)
smoothness_se = st.number_input('Smoothness Standard Error', min_value=0.001, max_value=0.03, value=0.001, step=0.01)
compactness_se = st.number_input('Compactness Standard Error', min_value=0.002, max_value=0.14, value=0.002, step=0.01)
concavity_se = st.number_input('Concavity Standard Error', min_value=0.00, max_value=0.400, value=0.00, step=0.01)
concave_points_se = st.number_input('Concave Points Standard Error', min_value=0.00, max_value=0.05, value=0.00, step=0.01)
symmetry_se = st.number_input('Symmetry Standard Error', min_value=0.007, max_value=0.08, value=0.007, step=0.01)
fractal_dimension_se = st.number_input('Fractal Dimension Standard Error', min_value=0.0009, max_value=0.03, value=0.0009, step=0.01)
radius_worst = st.number_input('Worst Radius', min_value=7, max_value=36, value=7, step=1)
texture_worst = st.number_input('Worst Texture', min_value=12, max_value=50, value=12, step=1)
perimeter_worst = st.number_input('Worst Perimeter', min_value=50, max_value=250, value=50, step=1)
area_worst = st.number_input('Worst Area', min_value=185, max_value=4250, value=185, step=1)
smoothness_worst = st.number_input('Worst Smoothness', min_value=0.07, max_value=0.22, value=0.07, step=0.01)
compactness_worst = st.number_input('Worst Compactness', min_value=0.03, max_value=1.06, value=0.03, step=0.01)
concavity_worst = st.number_input('Worst Concavity', min_value=0.00, max_value=1.25, value=0.00, step=0.01)
concave_points_worst = st.number_input('Worst Concave Points', min_value=0.00, max_value=0.29, value=0.00, step=0.01)
symmetry_worst = st.number_input('Worst Symmetry', min_value=0.16, max_value=0.66, value=0.16, step=0.01)
fractal_dimension_worst = st.number_input('Worst Fractal Dimension', min_value=0.05, max_value=0.21, value=0.05, step=0.01)


input_data = np.array([[radius_mean, texture_mean, perimeter_mean,area_mean, smoothness_mean, compactness_mean, concavity_mean,concave_points_mean, symmetry_mean, fractal_dimension_mean,radius_se, texture_se,perimeter_se, area_se, smoothness_se,compactness_se, concavity_se, concave_points_se, symmetry_se,fractal_dimension_se, radius_worst, texture_worst,perimeter_worst, area_worst, smoothness_worst,compactness_worst, concavity_worst, concave_points_worst,symmetry_worst, fractal_dimension_worst]])

if st.button('Diagnose'):
    prediction = model.predict(input_data)
    st.success(f'This Breast cancer is: {'Malignant' if prediction[0] == 1 else 'Beningn'}')

    st.write('Note: This diagnosis might not be accurate')
    st.write('Contact a health specialist to confirm the diagnosis  OR Visit a health center')
    st.write('by SCA')




