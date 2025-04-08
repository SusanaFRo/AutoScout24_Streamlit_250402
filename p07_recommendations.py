# =======================
# 1. Importieren von Bibliotheken
# =======================
import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =======================
# 2. Daten laden
# =======================

@st.cache_data
def load_data():
    df = pd.read_csv("autoscout24.csv")
    return df.dropna()

df = load_data()


# ---------------------
# VORAUSGESAGTE PREISE
# ---------------------
# Modelltraining für die Preisvorhersage
merkmale = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
X = df[merkmale]
y = df["price"]
X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
scaler = StandardScaler()
X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

@st.cache_resource
def daten_vorbereiten(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["make", "model", "fuel", "gear"], drop_first=True)
    for col in X_train.columns:
        if col not in df:
            df[col] = 0
    df = df[X_train.columns]
    df[["mileage", "hp", "year"]] = scaler.transform(df[["mileage", "hp", "year"]])
    return df


def autos_empfehlen(budget, df, preis_option, translate_text):
    trans_predicted_price = translate_text("Vorhergesagte Preise")
    trans_real_price = translate_text("Echte Preise")
    
    if preis_option == trans_predicted_price:
        df_verarbeitet = daten_vorbereiten(df[merkmale])
        df[trans_predicted_price] = xgb_model.predict(df_verarbeitet)
        autos = df[df[trans_predicted_price] <= budget].sort_values(by=trans_predicted_price, ascending=False)
    else:
        autos = df[df["price"] <= budget].sort_values(by="price", ascending=False)
    return autos


def app(translate_text):
    # Títulos y encabezados
    st.header("🚗 " + translate_text("7. Finde das beste Auto für dein Budget"))

    # Selección entre precios reales o predichos
    preis_option = st.selectbox(translate_text("Wählen Sie, ob Sie echte Preise oder vorhergesagte Preise möchten"), 
                                [translate_text("Vorhergesagte Preise"), translate_text("Echte Preise")])

    budget = st.number_input(translate_text("Gib dein Budget in EUR ein"), min_value=500, max_value=200000, value=10000, step=500, format="%d")
    
    if st.button("🔍 " + translate_text("Autos suchen")):
        ergebnisse = autos_empfehlen(budget, df, preis_option, translate_text)
        
        if ergebnisse.empty:
            st.warning(translate_text("Keine Autos innerhalb deines Budgets gefunden."))
        else:
            st.success(f"{translate_text('Es wurden')} {len(ergebnisse)} {translate_text('Autos innerhalb deines Budgets gefunden.')} 🚗💰")
            
            # Formateo para la visualización
            ergebnisse_anzeigen = ergebnisse.copy()
            if preis_option == translate_text("Vorhergesagte Preise"):
                ergebnisse_anzeigen[translate_text("Vorhergesagte Preise")] = ergebnisse_anzeigen[translate_text("Vorhergesagte Preise")].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                st.dataframe(ergebnisse_anzeigen[["make", "model", "year", "mileage", "fuel", "gear", "hp", translate_text("Vorhergesagte Preise")]])
            else:
                ergebnisse_anzeigen["price"] = ergebnisse_anzeigen["price"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                st.dataframe(ergebnisse_anzeigen[["make", "model", "year", "mileage", "fuel", "gear", "hp", "price"]])

            # Visualización de la distribución de precios
            if preis_option == translate_text("Vorhergesagte Preise"):
                fig = px.histogram(ergebnisse, x=translate_text("Vorhergesagte Preise"), title=translate_text("Preisverteilung innerhalb des Budgets"))
            else:
                fig = px.histogram(ergebnisse, x="price", title=translate_text("Preisverteilung innerhalb des Budgets"))
            st.plotly_chart(fig)

            # Texto de aclaración
            if preis_option == translate_text("Vorhergesagte Preise"):
                st.write(f"*{translate_text('Die Ergebnisse basieren auf Preisvorhersagen, die mit Hilfe von Machine Learning erstellt wurden.')}*")
            else:
                st.write(f"*{translate_text('Die Ergebnisse beruhen auf den tatsächlichen Preisen.')}*")

