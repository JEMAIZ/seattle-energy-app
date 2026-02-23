import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Configuration
st.set_page_config(page_title="Seattle Energy Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model_pipeline_seattle.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("2016_Building_Energy_Benchmarking.csv")

def main():
    st.sidebar.title("??? Navigation")
    page = st.sidebar.radio("Aller vers", ["Analyse Exploratoire", "Simulateur de Prédiction"])
    
    df = load_data()
    model = load_model()

    if page == "Analyse Exploratoire":
        st.title("?? Exploration des données de Seattle")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter(df, x="PropertyGFATotal", y="SiteEnergyUse(kBtu)", 
                             color="BuildingType", log_x=True, log_y=True,
                             title="Consommation vs Surface (Log)")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.box(df, x="BuildingType", y="SiteEnergyUse(kBtu)", log_y=True,
                         title="Répartition par Type de Bâtiment")
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.title("?? Simulateur de Consommation Énergétique")
        st.write("Entrez les caractéristiques d'un bâtiment pour estimer sa consommation annuelle.")

        with st.form("pred_form"):
            c1, c2 = st.columns(2)
            with c1:
                prop_type = st.selectbox("Type de propriété", df['PrimaryPropertyType'].unique())
                neighborhood = st.selectbox("Quartier", df['Neighborhood'].unique())
                gfa = st.number_input("Surface Totale (sq ft)", min_value=100, value=50000)
            with c2:
                floors = st.slider("Nombre d'étages", 1, 50, 5)
                age = st.number_input("Âge du bâtiment", 0, 150, 20)
                nb_bld = st.number_input("Nombre de bâtiments", 1, 10, 1)
            
            submit = st.form_submit_button("Lancer la prédiction")

        if submit:
            input_data = pd.DataFrame({
                'PrimaryPropertyType': [prop_type], 'Neighborhood': [neighborhood],
                'PropertyGFATotal': [gfa], 'NumberofFloors': [floors],
                'BuildingAge': [age], 'NumberofBuildings': [nb_bld]
            })
            
            # Prédiction et transformation inverse du Log
            prediction_log = model.predict(input_data)[0]
            prediction_final = np.expm1(prediction_log)
            
            st.success(f"### Consommation estimée : {prediction_final:,.2f} kBtu")
            st.info("Ce calcul est basé sur un modèle XGBoost entraîné sur les données réelles de 2016.")

if __name__ == "__main__":
    main()