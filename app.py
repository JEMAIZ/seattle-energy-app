# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Seattle Energy Predictor",
    page_icon="⚡",
    layout="wide",
)

DATA_PATH = os.path.join("data", "2016_Building_Energy_Benchmarking.csv")
MODEL_PATH = "model_pipeline_seattle.joblib"


# ── Chargement du modèle ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            "❌ Modèle introuvable. Veuillez lancer `train_model.py` d'abord."
        )
        st.stop()


# ── Chargement des données ───────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH, encoding="latin1")
    except FileNotFoundError:
        try:
            df = pd.read_csv(
                "2016_Building_Energy_Benchmarking.csv", encoding="latin1"
            )
        except FileNotFoundError:
            st.error("❌ Fichier de données introuvable.")
            st.stop()
    # ✅ On applique le même filtre que lors de l'entraînement
    df = df[df["SiteEnergyUse(kBtu)"] > 0].copy()
    df["BuildingAge"] = 2016 - df["YearBuilt"]
    return df


# ── Page : Analyse Exploratoire ──────────────────────────────────────────────
def page_exploration(df: pd.DataFrame) -> None:
    st.title("🔎 Exploration des données de Seattle")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            df,
            x="PropertyGFATotal",
            y="SiteEnergyUse(kBtu)",
            color="PrimaryPropertyType",   # ✅ correction (BuildingType inexistant)
            log_x=True,
            log_y=True,
            title="Consommation vs Surface (échelle log)",
            labels={
                "PropertyGFATotal": "Surface totale (sq ft)",
                "SiteEnergyUse(kBtu)": "Consommation (kBtu)",
            },
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(
            df,
            x="PrimaryPropertyType",       # ✅ correction (BuildingType inexistant)
            y="SiteEnergyUse(kBtu)",
            log_y=True,
            title="Répartition par Type de Bâtiment",
            labels={
                "PrimaryPropertyType": "Type de propriété",
                "SiteEnergyUse(kBtu)": "Consommation (kBtu)",
            },
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Statistiques rapides ─────────────────────────────────────────────────
    st.subheader("📊 Statistiques descriptives")
    st.dataframe(
        df[["SiteEnergyUse(kBtu)", "PropertyGFATotal", "NumberofFloors", "BuildingAge"]]
        .describe()
        .round(2),
        use_container_width=True,
    )


# ── Page : Simulateur de prédiction ─────────────────────────────────────────
def page_prediction(df: pd.DataFrame, model) -> None:
    st.title("⚡ Simulateur de Consommation Énergétique")
    st.write(
        "Entrez les caractéristiques d'un bâtiment pour estimer sa consommation annuelle."
    )

    with st.form("pred_form"):
        c1, c2 = st.columns(2)

        with c1:
            prop_type = st.selectbox(
                "Type de propriété",
                sorted(df["PrimaryPropertyType"].dropna().unique()),
            )
            neighborhood = st.selectbox(
                "Quartier",
                sorted(df["Neighborhood"].dropna().unique()),
            )
            gfa = st.number_input(
                "Surface totale (sq ft)", min_value=100, max_value=5_000_000, value=50_000, step=500
            )

        with c2:
            floors = st.slider("Nombre d'étages", 1, 50, 5)
            age = st.number_input("Âge du bâtiment (années)", 0, 200, 20)
            nb_bld = st.number_input("Nombre de bâtiments", 1, 100, 1)

        submit = st.form_submit_button("🚀 Lancer la prédiction")

    if submit:
        input_data = pd.DataFrame({
            "PrimaryPropertyType": [prop_type],
            "Neighborhood":        [neighborhood],
            "PropertyGFATotal":    [gfa],
            "NumberofFloors":      [floors],
            "BuildingAge":         [age],
            "NumberofBuildings":   [nb_bld],
        })

        # ✅ Transformation inverse du log (np.log1p → np.expm1)
        prediction_log   = model.predict(input_data)[0]
        prediction_final = np.expm1(prediction_log)

        st.success(f"### 🏢 Consommation estimée : {prediction_final:,.0f} kBtu/an")

        # ── Mise en contexte ─────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("kBtu / an",       f"{prediction_final:,.0f}")
        col_b.metric("kWh équivalent",  f"{prediction_final / 3.412:,.0f}")
        col_c.metric("kBtu / sq ft",    f"{prediction_final / gfa:.1f}")

        st.info(
            "ℹ️ Ce calcul est basé sur un modèle XGBoost entraîné sur les données "
            "réelles de benchmarking énergétique de Seattle (2016)."
        )

        # ── Feature importance (top 10) ──────────────────────────────────────
        with st.expander("📈 Voir l'importance des variables du modèle"):
            regressor = model.named_steps["regressor"]
            preprocessor = model.named_steps["preprocessor"]

            # Récupération des noms de features après encodage
            num_names = preprocessor.transformers_[0][2]
            cat_names = (
                preprocessor.transformers_[1][1]
                .named_steps["onehot"]
                .get_feature_names_out(preprocessor.transformers_[1][2])
                .tolist()
            )
            all_names = num_names + cat_names

            importances = regressor.feature_importances_
            fi_df = (
                pd.DataFrame({"Feature": all_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(10)
            )
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 variables les plus importantes",
            )
            fig_fi.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── Upload batch ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📂 Prédictions en lot (CSV)")
    uploaded = st.file_uploader(
        "Uploadez un CSV avec les colonnes : PrimaryPropertyType, Neighborhood, "
        "PropertyGFATotal, NumberofFloors, BuildingAge, NumberofBuildings",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            batch_df   = pd.read_csv(uploaded, encoding="latin1")
            preds_log  = model.predict(batch_df)
            preds_kbtu = np.expm1(preds_log)
            result_df  = batch_df.copy()
            result_df["Prédiction (kBtu)"] = preds_kbtu.round(0)
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "⬇️ Télécharger les résultats",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions_seattle.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.sidebar.title("🏙️ Seattle Energy App")
    page = st.sidebar.radio(
        "Navigation",
        ["🔎 Analyse Exploratoire", "⚡ Simulateur de Prédiction"],
    )

    df    = load_data()
    model = load_model()

    if page == "🔎 Analyse Exploratoire":
        page_exploration(df)
    else:
        page_prediction(df, model)


if __name__ == "__main__":
    main()
