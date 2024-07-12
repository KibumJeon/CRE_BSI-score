import pandas as pd
import numpy as np
import pickle
import shap
import streamlit as st
import streamlit.components.v1 as components

model_path = "/Users/kj/Library/CloudStorage/OneDrive-한림대학교/연구 및 과제/박사논문/after14/model.pickle.dat"
model = pickle.load(open(model_path, "rb"))
explainer = shap.TreeExplainer(model)

display_name = pd.read_excel("/Users/kj/Library/CloudStorage/OneDrive-한림대학교/연구 및 과제/박사논문/after14/feature_list_Display_name after14.xlsx")


def predict(features):
    shap_values = explainer.shap_values(features)
    log_odds = model.predict(features, output_margin=True)
    probability = 1 / (1 + np.exp(-log_odds))
    return shap_values, probability


st.markdown('<span style="font-size: 40px;">Prediction of bloodstream infection within 14 days after positive rectal CRE surveillance culture</span><span style="font-size: 18px;">&nbsp;&nbsp;&nbsp; by K.Jeon</span>', unsafe_allow_html=True)


cols = st.columns(2)
input_data = {}
idx = 0

for _, row in display_name.iterrows():
    with cols[idx % 2]:
        # 초기값을 None으로 설정
        user_input = st.number_input(row['Display_Name'], format="%f", value=None, step=1.0, key=row['Type_Short_Name'])
        input_data[row['Type_Short_Name']] = user_input
    idx += 1


features = pd.DataFrame([input_data])
features = features.fillna(0).astype(float)  


if any(value is not None for value in input_data.values()):  
    # 입력 데이터를 DataFrame으로 변환
    features = pd.DataFrame([input_data])
    features = features.fillna(0).astype(float)  

    shap_values, probability = predict(features)


    st.write(f'Prediction probability: {probability[0] * 100:.1f}%')

    if 'shap_values' in locals():
        shap_html = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            features.iloc[0], 
            matplotlib=False, 
            show=False, 
            feature_names=[row['short_name'] for _, row in display_name.iterrows() if row['Type_Short_Name'] in features.columns]
        )
        shap_html_str = f"<head>{shap.getjs()}</head><body>{shap_html.html()}</body>"
        components.html(shap_html_str, height=200, width=1200)
