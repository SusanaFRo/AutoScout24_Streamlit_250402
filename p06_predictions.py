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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =======================
# 2. Daten laden
# =======================
#df = pd.read_csv('autoscout24.csv', sep=',', decimal='.')

@st.cache_data
def load_data():
    df = pd.read_csv("autoscout24.csv")
    return df.dropna()

df = load_data()


def app(translate_text):


    # =======================
    # Auswahl der Merkmale
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Kategorische Variablen kodieren
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Numerische Variablen normalisieren
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modell trainieren
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Streamlit-Oberfläche
    st.header("🚗 " + translate_text("6. Autokonfigurator: Preisvorhersage"))

    
    translated_mark = translate_text("Marke")
    translated_model = translate_text("Modell")
    translated_fuel = translate_text("Kraftstoff")
    translated_gear = translate_text("Getriebe")
    translated_mileage = translate_text("Kilometerstand (km)")
    translated_year = translate_text("Jahr")
    
    make = st.selectbox(translated_mark, df["make"].unique())
    model = st.selectbox(translated_model, df[df["make"] == make]["model"].unique())
    fuel = st.selectbox(translated_fuel, df["fuel"].unique())
    gear = st.selectbox(translated_gear, df["gear"].unique())

    
    # Mindest- und Höchstwerte ermitteln
    min_km, max_km = int(df["mileage"].min()), int(df["mileage"].max())  # In Int konvertieren
    min_hp, max_hp = int(df["hp"].min()), int(df["hp"].max())  # In Int konvertieren
    min_year, max_year = int(df["year"].min()), int(df["year"].max())  # In Int konvertieren
    
    # Festgelegte Grenzen
    max_km_limit = max_km * 5
    max_hp_limit = 10000
    max_year_limit = max_year + 50

    
    # Auswahl-Widgets mit begrenzten Werten
    mileage = st.number_input(
        f"{translated_mileage} (Min: {min_km}, Max: {max_km})",
        min_value=min_km, 
        max_value=max_km_limit, 
        value=min_km, 
        step=1,  # Nur ganze Zahlen
        help=f"Min: {min_km}, Max: {max_km_limit}"
    )
    
    hp = st.number_input(
        f"PS (Min: {min_hp}, Max: {max_hp})",
        min_value=min_hp, 
        max_value=max_hp_limit, 
        value=min_hp, 
        step=1,  # Nur ganze Zahlen
        help=f"Min: {min_hp}, Max: {max_hp_limit}"
    )
    
    year = st.number_input(
        f"{translated_year} (Min: {min_year}, Max: {max_year})",
        min_value=min_year, 
        max_value=max_year_limit, 
        value=max_year, 
        step=1,  # Nur ganze Zahlen
        help=f"Min: {min_year}, Max: {max_year_limit}"
    )
    
    # Bedingung, um den DataFrame zu erstellen und die Vorhersage nur auszuführen, wenn der Button gedrückt wird
    if st.button("🔍 " + translate_text("Preis vorhersagen")):
        # Eingabedatenrahmen erstellen
        car_input = pd.DataFrame({
            "mileage": [mileage],
            "hp": [hp],
            "year": [year],
            **{f"make_{make}": [1]},
            **{f"model_{model}": [1]},
            **{f"fuel_{fuel}": [1]},
            **{f"gear_{gear}": [1]}
        })
        
        # Sicherstellen, dass alle Spalten mit X_train übereinstimmen
        for col in X_train.columns:
            if col not in car_input:
                car_input[col] = 0
        
        # Spalten neu anordnen und Werte normalisieren
        car_input = car_input[X_train.columns]
        car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])
        
        # Vorhersage
        predicted_price = xgb_model.predict(car_input)[0]

        # Preis formatieren (Tausender mit Punkt, Dezimal mit Komma)
        predicted_price_str = f"{predicted_price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        
        # Vorhergesagten Preis anzeigen
        st.success(f"💰 {translate_text("Der geschätzte Preis beträgt: ")}{predicted_price_str} EUR")

        st.write("*" + translate_text("Die Ergebnisse basieren auf Preisvorhersagen, die mit Hilfe von Machine Learning erstellt wurden.") + "*")

    