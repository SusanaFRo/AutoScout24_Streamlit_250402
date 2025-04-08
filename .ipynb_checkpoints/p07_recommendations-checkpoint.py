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

@st.cache_resource
def autos_empfehlen(budget, df, preis_option):
    if preis_option == "Vorhergesagte Preise":
        df_verarbeitet = daten_vorbereiten(df[merkmale])
        df["preis_vorhersage"] = xgb_model.predict(df_verarbeitet)
        autos = df[df["preis_vorhersage"] <= budget].sort_values(by="preis_vorhersage", ascending=False)
    else:
        autos = df[df["price"] <= budget].sort_values(by="price", ascending=False)
    return autos


def app():
    st.header("🚗 7. Finde das beste Auto für dein Budget")

    # Selección entre precios reales o predichos
    preis_option = st.selectbox("Wählen Sie, ob Sie echte Preise oder vorhergesagte Preise möchten", 
                                ["Echte Preise", "Vorhergesagte Preise"])

    budget = st.number_input("Gib dein Budget in EUR ein", min_value=500, max_value=200000, value=10000, step=500, format="%d")
    
    if st.button("🔍 Autos suchen"):
        ergebnisse = autos_empfehlen(budget, df, preis_option)
        
        if ergebnisse.empty:
            st.warning("Keine Autos innerhalb deines Budgets gefunden.")
        else:
            st.success(f" Es wurden {len(ergebnisse)} Autos 🚗 innerhalb deines Budgets 💰 gefunden.")
            
            # Formatierung für die Anzeige
            ergebnisse_anzeigen = ergebnisse.copy()
            if preis_option == "Vorhergesagte Preise":
                ergebnisse_anzeigen["preis_vorhersage"] = ergebnisse_anzeigen["preis_vorhersage"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                st.dataframe(ergebnisse_anzeigen[["make", "model", "year", "mileage", "fuel", "gear", "hp", "preis_vorhersage"]])
            else:
                ergebnisse_anzeigen["price"] = ergebnisse_anzeigen["price"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                st.dataframe(ergebnisse_anzeigen[["make", "model", "year", "mileage", "fuel", "gear", "hp", "price"]])

            # Visualisierung der Preisverteilung
            if preis_option == "Vorhergesagte Preise":
                fig = px.histogram(ergebnisse, x="preis_vorhersage", title="Preisverteilung innerhalb des Budgets")
            else:
                fig = px.histogram(ergebnisse, x="price", title="Preisverteilung innerhalb des Budgets")
            st.plotly_chart(fig)

            st.write("""*Die Ergebnisse basieren auf Preisvorhersagen, die mit Hilfe von Machine Learning erstellt wurden.*""") if preis_option == "Vorhergesagte Preise" else st.write("""*Die Ergebnisse beruhen auf den tatsächlichen Preisen.*""")
    





