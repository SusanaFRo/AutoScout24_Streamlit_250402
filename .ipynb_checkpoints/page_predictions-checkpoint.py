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



def app():


    st.header("7. Prognosen")
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
    st.title("Autokonfigurator: Preisvorhersage")
    
    make = st.selectbox("Marke", df["make"].unique())
    model = st.selectbox("Modell", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    gear = st.selectbox("Getriebe", df["gear"].unique())
    
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
        f"Kilometerstand (Min: {min_km}, Max: {max_km})",
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
        f"Baujahr (Min: {min_year}, Max: {max_year})",
        min_value=min_year, 
        max_value=max_year_limit, 
        value=max_year, 
        step=1,  # Nur ganze Zahlen
        help=f"Min: {min_year}, Max: {max_year_limit}"
    )
    
    # Bedingung, um den DataFrame zu erstellen und die Vorhersage nur auszuführen, wenn der Button gedrückt wird
    if st.button("Preis vorhersagen"):
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
        
        # Vorhergesagten Preis anzeigen
        st.success(f"Der geschätzte Preis beträgt: {predicted_price:.2f} EUR")

    # =======================
    '''
    # =======================
    
    
    # Selección de características
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Codificar variables categóricas
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Interfaz en Streamlit
    st.title("Autokonfigurator: Preisvorhersage")

    make = st.selectbox("Marke", df["make"].unique())
    model = st.selectbox("Modell", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    gear = st.selectbox("Getriebe", df["gear"].unique())

    # Obtener valores mínimos y máximos
    min_km, max_km = int(df["mileage"].min()), int(df["mileage"].max())  # Convertir a int
    min_hp, max_hp = int(df["hp"].min()), int(df["hp"].max())  # Convertir a int
    min_year, max_year = int(df["year"].min()), int(df["year"].max())  # Convertir a int
    
    # Limites establecidos
    max_km_limit = max_km * 5
    max_hp_limit = 10000
    max_year_limit = max_year + 50

    # Widgets de selección con valores limitados
    mileage = st.number_input(
        #"Kilometerstand (Km)",
        f"Kilometraje (Min: {min_km}, Max: {max_km})",
        min_value=min_km, 
        max_value=max_km_limit, 
        value=min_km, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_km}, Max: {max_km_limit}"
    )
    
    hp = st.number_input(
        #"Leistung (PS)",
        f"PS (Min: {min_hp}, Max: {max_hp})",
        min_value=min_hp, 
        max_value=max_hp_limit, 
        value=min_hp, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_hp}, Max: {max_hp_limit}"
    )
    
    year = st.number_input(
        #"Baujahr",
        f"Baujahr (Min: {min_year}, Max: {max_year})",
        min_value=min_year, 
        max_value=max_year_limit, 
        value=max_year, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_year}, Max: {max_year_limit}"
    )
    
    # Condicional para crear el DataFrame y hacer la predicción solo cuando se presiona el botón
    if st.button("Preis vorhersagen"):
        # Construir DataFrame de entrada
        car_input = pd.DataFrame({
            "mileage": [mileage],
            "hp": [hp],
            "year": [year],
            **{f"make_{make}": [1]},
            **{f"model_{model}": [1]},
            **{f"fuel_{fuel}": [1]},
            **{f"gear_{gear}": [1]}
        })
    
        # Asegurarse de que todas las columnas coincidan con X_train
        for col in X_train.columns:
            if col not in car_input:
                car_input[col] = 0
    
        # Reordenar columnas y normalizar valores
        car_input = car_input[X_train.columns]
        car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])
    
        # Predicción
        predicted_price = xgb_model.predict(car_input)[0]
        
        # Mostrar el precio predicho
        st.success(f"Der geschätzte Preis beträgt: {predicted_price:.2f} EUR")
    # =======================
    # =======================
    
    # =======================
    
    # Obtener valores mínimos y máximos
    min_km, max_km = int(df["mileage"].min()), int(df["mileage"].max())  # Convertir a int
    min_hp, max_hp = int(df["hp"].min()), int(df["hp"].max())  # Convertir a int
    min_year, max_year = int(df["year"].min()), int(df["year"].max())  # Convertir a int
    
    # Limites establecidos
    max_km_limit = max_km * 5
    max_hp_limit = 10000
    max_year_limit = max_year + 50
    
    # Selección de características
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Codificar variables categóricas
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Interfaz en Streamlit
    st.title("Autokonfigurator: Preisvorhersage")
    
    make = st.selectbox("Marke (Marca)", df["make"].unique())
    model = st.selectbox("Modell (Modelo)", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff (Combustible)", df["fuel"].unique())
    gear = st.selectbox("Getriebe (Transmisión)", df["gear"].unique())
    
    # Mostrar los valores mínimos y máximos
    st.sidebar.header("Eingabebereich für numerische Merkmale:")
    st.sidebar.write(f"Kilometerstand: Min = {min_km}, Max = {max_km_limit}")
    st.sidebar.write(f"PS (Leistung): Min = {min_hp}, Max = {max_hp_limit}")
    st.sidebar.write(f"Baujahr: Min = {min_year}, Max = {max_year_limit}")
    
    # Widgets de selección con valores limitados
    mileage = st.number_input(
        "Kilometerstand (Kilómetros)", 
        min_value=min_km, 
        max_value=max_km_limit, 
        value=min_km, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_km}, Max: {max_km_limit}"
    )
    
    hp = st.number_input(
        "PS (Leistung)", 
        min_value=min_hp, 
        max_value=max_hp_limit, 
        value=min_hp, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_hp}, Max: {max_hp_limit}"
    )
    
    year = st.number_input(
        "Baujahr (Año)", 
        min_value=min_year, 
        max_value=max_year_limit, 
        value=max_year, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_year}, Max: {max_year_limit}"
    )
    
    # Condicional para crear el DataFrame y hacer la predicción solo cuando se presiona el botón
    if st.button("Preis vorhersagen"):
        # Construir DataFrame de entrada
        car_input = pd.DataFrame({
            "mileage": [mileage],
            "hp": [hp],
            "year": [year],
            **{f"make_{make}": [1]},
            **{f"model_{model}": [1]},
            **{f"fuel_{fuel}": [1]},
            **{f"gear_{gear}": [1]}
        })
    
        # Asegurarse de que todas las columnas coincidan con X_train
        for col in X_train.columns:
            if col not in car_input:
                car_input[col] = 0
    
        # Reordenar columnas y normalizar valores
        car_input = car_input[X_train.columns]
        car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])
    
        # Predicción
        predicted_price = xgb_model.predict(car_input)[0]
        
        # Mostrar el precio predicho
        st.success(f"Der geschätzte Preis beträgt: {predicted_price:.2f} EUR")

    # =======================
   
    # =======================


    # Obtener valores mínimos y máximos
    min_km, max_km = int(df["mileage"].min()), int(df["mileage"].max())  # Convertir a int
    min_hp, max_hp = int(df["hp"].min()), int(df["hp"].max())  # Convertir a int
    min_year, max_year = int(df["year"].min()), int(df["year"].max())  # Convertir a int
    
    # Limites establecidos
    max_km_limit = max_km * 5
    max_hp_limit = 10000
    max_year_limit = max_year + 50
    
    # Selección de características
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Codificar variables categóricas
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Interfaz en Streamlit
    st.title("Autokonfigurator: Preisvorhersage")

    make = st.selectbox("Marke (Marca)", df["make"].unique())
    model = st.selectbox("Modell (Modelo)", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff (Combustible)", df["fuel"].unique())
    gear = st.selectbox("Getriebe (Transmisión)", df["gear"].unique())
    
    # Mostrar los valores mínimos y máximos
    st.sidebar.header("Eingabebereich für numerische Merkmale:")
    st.sidebar.write(f"Kilometerstand: Min = {min_km}, Max = {max_km_limit}")
    st.sidebar.write(f"PS (Leistung): Min = {min_hp}, Max = {max_hp_limit}")
    st.sidebar.write(f"Baujahr: Min = {min_year}, Max = {max_year_limit}")


    # Widgets de selección con valores limitados
    mileage = st.number_input(
        "Kilometerstand (Kilómetros)", 
        min_value=min_km, 
        max_value=max_km_limit, 
        value=min_km, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_km}, Max: {max_km_limit}"
    )
    
    hp = st.number_input(
        "PS (Leistung)", 
        min_value=min_hp, 
        max_value=max_hp_limit, 
        value=min_hp, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_hp}, Max: {max_hp_limit}"
    )
    
    year = st.number_input(
        "Baujahr (Año)", 
        min_value=min_year, 
        max_value=max_year_limit, 
        value=max_year, 
        step=1,  # Solo valores enteros
        help=f"Min: {min_year}, Max: {max_year_limit}"
    )


    # Botón para hacer la predicción
    if st.button("Preis vorhersagen"):
        # Construir DataFrame de entrada
        car_input = pd.DataFrame({
            "mileage": [mileage],
            "hp": [hp],
            "year": [year],
            **{f"make_{make}": [1]},
            **{f"model_{model}": [1]},
            **{f"fuel_{fuel}": [1]},
            **{f"gear_{gear}": [1]}
        })

    # Asegurarse de que todas las columnas coincidan con X_train
    for col in X_train.columns:
        if col not in car_input:
            car_input[col] = 0

    # Reordenar columnas y normalizar valores
    car_input = car_input[X_train.columns]
    car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])

    # Predicción
    predicted_price = xgb_model.predict(car_input)[0]
    
    # Mostrar el precio predicho
    st.success(f"Der geschätzte Preis beträgt: {predicted_price:.2f} EUR")
    
    # =======================
 
    # =======================
    
    # Obtener valores mínimos y máximos
    min_km, max_km = df["mileage"].min(), df["mileage"].max()
    min_hp, max_hp = df["hp"].min(), df["hp"].max()
    min_year, max_year = df["year"].min(), df["year"].max()
    
    # Limites establecidos
    max_km_limit = max_km * 5
    max_hp_limit = 10000
    max_year_limit = max_year + 50
    
    # Selección de características
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Codificar variables categóricas
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Interfaz en Streamlit
    st.title("Autokonfigurator: Preisvorhersage")
    
    # Widgets de selección
    mileage = st.number_input("Kilometerstand", min_value=min_km, max_value=max_km_limit, value=min_km, help=f"Min: {min_km}, Max: {max_km_limit}")
    hp = st.number_input("PS (Leistung)", min_value=min_hp, max_value=max_hp_limit, value=min_hp, help=f"Min: {min_hp}, Max: {max_hp_limit}")
    year = st.number_input("Baujahr", min_value=min_year, max_value=max_year_limit, value=max_year, help=f"Min: {min_year}, Max: {max_year_limit}")
    make = st.selectbox("Marke", df["make"].unique())
    model = st.selectbox("Modell", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    gear = st.selectbox("Getriebe", df["gear"].unique())
    
    if st.button("Preis vorhersagen"):
        # Construir DataFrame de entrada
        car_input = pd.DataFrame({
            "mileage": [mileage],
            "hp": [hp],
            "year": [year],
            **{f"make_{make}": [1]},
            **{f"model_{model}": [1]},
            **{f"fuel_{fuel}": [1]},
            **{f"gear_{gear}": [1]}
        })
    
        # Asegurar que todas las columnas coincidan con X_train
        for col in X_train.columns:
            if col not in car_input:
                car_input[col] = 0
    
        # Reordenar columnas y normalizar valores
        car_input = car_input[X_train.columns]
        car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])
    
        # Predicción
        predicted_price = xgb_model.predict(car_input)[0]
        st.success(f"Der geschätzte Preis beträgt: {predicted_price:.2f} EUR")

    
    # =======================
   
    # =======================
    
 

    
    # Selección de características
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Obtener valores únicos para menús desplegables
    makes = df["make"].unique()
    models = df["model"].unique()
    fuel_types = df["fuel"].unique()
    gear_types = df["gear"].unique()
    
    # Obtener rangos de valores para inputs numéricos
    mileage_min, mileage_max = df["mileage"].min(), df["mileage"].max()
    hp_min, hp_max = df["hp"].min(), df["hp"].max()
    year_min, year_max = df["year"].min(), df["year"].max()
    
    # Codificación de variables categóricas
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Normalización de variables numéricas
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo XGBoost
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Interfaz en Streamlit
    st.title("🚗 AutoScout24 - Predicción de Precio del Coche")
    
    # Selección de características
    t_make = st.selectbox("Marca", makes)
    t_model = st.selectbox("Modelo", models)
    t_fuel = st.selectbox("Combustible", fuel_types)
    t_gear = st.selectbox("Transmisión", gear_types)
    
    t_mileage = st.number_input(f"Kilometraje (Min: {mileage_min}, Max: {mileage_max})", value=mileage_min)
    t_hp = st.number_input(f"PS (Min: {hp_min}, Max: {hp_max})", value=hp_min)
    t_year = st.number_input(f"Año (Min: {year_min}, Max: 2035)", value=year_max, min_value=year_min, max_value=2035)
    
    # Crear DataFrame de entrada
    car_input = pd.DataFrame({
        "mileage": [t_mileage],
        "hp": [t_hp],
        "year": [t_year]
    })
    
    # Agregar columnas categóricas
    for col in X_train.columns:
        if col.startswith("make_"):
            car_input[col] = 1 if f"make_{t_make}" == col else 0
        elif col.startswith("model_"):
            car_input[col] = 1 if f"model_{t_model}" == col else 0
        elif col.startswith("fuel_"):
            car_input[col] = 1 if f"fuel_{t_fuel}" == col else 0
        elif col.startswith("gear_"):
            car_input[col] = 1 if f"gear_{t_gear}" == col else 0
        else:
            if col not in car_input:
                car_input[col] = 0
    
    # Reordenar columnas para coincidir con X_train
    car_input = car_input[X_train.columns]
    
    # Normalizar entrada
    car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])
    
    # Predicción
    if st.button("Predecir Precio"):
        predicted_price = xgb_model.predict(car_input)[0]
        st.success(f"💰 Precio Estimado: {predicted_price:.2f} EUR")

    
    # =======================

    # =======================

    # Merkmale und Zielvariable auswählen
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Kategorische Variablen kodieren
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Normalisierung
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Modell trainieren
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Wertebereich ermitteln
    min_mileage, max_mileage = df["mileage"].min(), df["mileage"].max()
    min_hp, max_hp = df["hp"].min(), df["hp"].max()
    min_year, max_year = df["year"].min(), df["year"].max()
    
    # Streamlit App
    st.title("Autokonfigurator und Preisprognose")
    
    # Auswahlboxen für kategorische Werte
    selected_make = st.selectbox("Marke", df["make"].unique())
    selected_model = st.selectbox("Modell", df[df["make"] == selected_make]["model"].unique())
    selected_fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    selected_gear = st.selectbox("Getriebe", df["gear"].unique())
    
    # Eingabefelder für numerische Werte mit Min/Max-Hinweisen
    mileage = st.number_input(f"Kilometerstand eingeben ({min_mileage}-{max_mileage})", min_value=0)
    hp = st.number_input(f"PS eingeben ({min_hp}-{max_hp})", min_value=0)
    year = st.number_input(f"Baujahr eingeben ({min_year}-{max_year}+)", min_value=2000)
    
    # Vorbereitung der Eingaben für das Modell
    car_input = pd.DataFrame({
        "mileage": [mileage],
        "hp": [hp],
        "year": [year]
    })
    
    # Kategorische Werte kodieren
    for col in X.columns:
        if col not in car_input.columns:
            car_input[col] = 0
    car_input[f"make_{selected_make}"] = 1
    car_input[f"model_{selected_model}"] = 1
    car_input[f"fuel_{selected_fuel}"] = 1
    car_input[f"gear_{selected_gear}"] = 1
    
    # Normalisierung mit demselben Scaler
    car_input[["mileage", "hp", "year"]] = scaler.transform(car_input[["mileage", "hp", "year"]])
    
    # Extrapolation für zukünftige Jahre
    if year > max_year:
        car_input["year"] = (year - min_year) / (max_year - min_year)  # Anpassung für Extrapolation
    
    # Preisprognose
    predicted_price = xgb_model.predict(car_input)[0]
    st.write(f"### Vorhergesagter Preis: {predicted_price:.2f} EUR")


    # =======================
  
    # =======================
    # =======================
    # Features definieren
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Kategorische Variablen kodieren
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Numerische Variablen normalisieren
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modell trainieren
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Streamlit-App starten
    st.title("Autokonfigurator & Preisvorhersage")
    
    # Dropdown-Menüs für die Auswahl der Fahrzeugmerkmale
    make = st.selectbox("Marke", df["make"].unique())
    model = st.selectbox("Modell", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    gear = st.selectbox("Getriebe", df["gear"].unique())
    
    # Eingabefelder für numerische Werte
    mileage = st.number_input("Kilometerstand", min_value=0, value=50000, step=5000)
    hp = st.number_input("PS", min_value=int(df["hp"].min()), max_value=int(df["hp"].max()), value=100, step=10)
    year = st.number_input("Baujahr", min_value=int(df["year"].min()), value=2021, step=1)
    
    # Sicherstellen, dass der Nutzer Jahre nach 2021 wählen kann
    if year > 2021:
        st.warning("Beachte: Das Modell wurde nur mit Daten bis 2021 trainiert. Vorhersagen für zukünftige Jahre sind extrapoliert!")
    
    # Vorhersage-Button
    test_input = pd.DataFrame({
        "mileage": [mileage],
        f"make_{make}": [1],
        f"model_{model}": [1],
        f"fuel_{fuel}": [1],
        f"gear_{gear}": [1],
        "hp": [hp],
        "year": [year]
    })
    
    # Fehlende Spalten hinzufügen
    test_input = pd.concat([test_input, pd.DataFrame({col: [0] for col in set(X.columns) - set(test_input.columns)})], axis=1)
    
    # Spaltenreihenfolge anpassen
    test_input = test_input[X.columns]
    
    # Normalisierung
    test_input[["mileage", "hp", "year"]] = scaler.transform(test_input[["mileage", "hp", "year"]])
    
    if st.button("Preis vorhersagen"):
        predicted_price = xgb_model.predict(test_input)[0]
        st.success(f"Der geschätzte Preis beträgt: {predicted_price:.2f} EUR")


    # =======================
    # =======================

 
    # Merkmale definieren
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Kategorische Variablen kodieren
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Numerische Variablen normalisieren
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost-Modell trainieren
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Streamlit-App
    st.title("Autokonfigurator - Preisvorhersage")
    
    # Dropdowns für kategorische Variablen
    selected_make = st.selectbox("Marke", df["make"].unique())
    selected_model = st.selectbox("Modell", df[df["make"] == selected_make]["model"].unique())
    selected_fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    selected_gear = st.selectbox("Getriebe", df["gear"].unique())
    
    # Eingabefelder für numerische Variablen
    selected_mileage = st.number_input("Kilometerstand", min_value=0, value=50000, step=1000)
    selected_hp = st.number_input("PS", min_value=int(df["hp"].min()), max_value=int(df["hp"].max()), value=100)
    selected_year = st.number_input("Baujahr", min_value=2011, max_value=2025, value=2021, step=1)
    
    # Vorhersage durchführen, wenn der Nutzer auf den Button klickt
    if st.button("Preis vorhersagen"):
        # Datensatz für die Vorhersage erstellen
        car_to_predict = pd.DataFrame({
            'mileage': [selected_mileage],
            'hp': [selected_hp],
            'year': [selected_year],
            f'make_{selected_make}': [1],
            f'model_{selected_model}': [1],
            f'fuel_{selected_fuel}': [1],
            f'gear_{selected_gear}': [1]
        })
        
    # Fehlende Spalten hinzufügen
    missing_cols = set(X.columns) - set(car_to_predict.columns)
    for col in missing_cols:
        car_to_predict[col] = 0
    
    # Spalten sortieren
    car_to_predict = car_to_predict[X.columns]
    
    # Numerische Variablen normalisieren
    car_to_predict[["mileage", "hp", "year"]] = scaler.transform(car_to_predict[["mileage", "hp", "year"]])
    
    # Preis vorhersagen
    predicted_price = xgb_model.predict(car_to_predict)
    
    # Ergebnis anzeigen
    st.success(f"Der geschätzte Preis beträgt: {predicted_price[0]:,.2f} EUR")

    
    # =======================
    
    # Merkmale auswählen
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Kategorische Variablen kodieren
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Numerische Variablen normalisieren
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Daten in Trainings- und Testset aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost-Modell erstellen und trainieren
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Streamlit App
    st.title("🚗 Autokonfigurator - Preisvorhersage")
    
    # Dropdowns für kategorische Merkmale
    make = st.selectbox("Marke", df["make"].unique())
    model = st.selectbox("Modell", df[df["make"] == make]["model"].unique())
    fuel = st.selectbox("Kraftstoff", df["fuel"].unique())
    gear = st.selectbox("Getriebe", df["gear"].unique())
    
    # Eingabefelder für numerische Werte
    mileage = st.number_input("Kilometerstand", min_value=0, value=int(df["mileage"].mean()))
    hp = st.number_input("Leistung (PS)", min_value=0, value=int(df["hp"].mean()))
    year = st.number_input("Baujahr", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=2012)
    
    # Vorhersage starten
    if st.button("🔍 Preis vorhersagen"):
        car_to_predict = pd.DataFrame({
            'mileage': [mileage],
            f'make_{make}': [1],
            f'model_{model}': [1],
            f'fuel_{fuel}': [1],
            f'gear_{gear}': [1],
            'hp': [hp],
            'year': [year],
        })
        
        # Fehlende Spalten hinzufügen
        missing_cols = set(X.columns) - set(car_to_predict.columns)
        car_to_predict = pd.concat([car_to_predict, pd.DataFrame({col: [0] for col in missing_cols})], axis=1)
        car_to_predict = car_to_predict[X.columns]
        
        # Normalisierung anwenden
        car_to_predict[["mileage", "hp", "year"]] = scaler.transform(car_to_predict[["mileage", "hp", "year"]])
        
        # Preisvorhersage
        predicted_price = xgb_model.predict(car_to_predict)
        st.success(f'📢 Der geschätzte Preis für Ihr Auto beträgt: **{predicted_price[0]:,.2f} EUR**')
    
        
    
# =======================
# =======================
    # Obtener valores únicos para las listas desplegables
    makes = df["make"].unique().tolist()
    models = df["model"].unique().tolist()
    fuel_types = df["fuel"].unique().tolist()
    gear_types = df["gear"].unique().tolist()
    
    # Crear los widgets en Streamlit
    st.title("Autokonfigurator")
    
    selected_make = st.selectbox("Marke auswählen", makes)
    selected_model = st.selectbox("Modell auswählen", models)
    selected_fuel = st.selectbox("Kraftstoffart auswählen", fuel_types)
    selected_gear = st.selectbox("Getriebe auswählen", gear_types)
    
    mileage = st.number_input("Kilometerstand eingeben", min_value=0, step=500, value=int(df["mileage"].mean()))
    hp = st.number_input("Leistung (PS) eingeben", min_value=0, step=5, value=int(df["hp"].mean()))
    year = st.number_input("Baujahr eingeben", min_value=int(df["year"].min()), max_value=int(df["year"].max()), step=1, value=int(df["year"].mean()))
    
    # Mostrar los valores seleccionados
    st.write("### Ausgewählte Fahrzeugdaten:")
    st.write(f"- **Marke:** {selected_make}")
    st.write(f"- **Modell:** {selected_model}")
    st.write(f"- **Kraftstoff:** {selected_fuel}")
    st.write(f"- **Getriebe:** {selected_gear}")
    st.write(f"- **Kilometerstand:** {mileage} km")
    st.write(f"- **Leistung:** {hp} PS")
    st.write(f"- **Baujahr:** {year}")



# Auswahl relevanter Merkmale
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = df[features]
    y = df["price"]
    
    # Kategorische Variablen kodieren
    X = pd.get_dummies(X, columns=["make", "model", "fuel", "gear"], drop_first=True)
    
    # Numerische Variablen normalisieren
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Aufteilen in Trainings- und Testdatensatz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost-Modell erstellen und trainieren
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Vorhersagen treffen
    y_pred = xgb_model.predict(X_test)
    
    # Modell evaluieren
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    # ------------------------------
    # 📌 Frage 1: Was wird der Preis eines Autos basierend auf Kilometerstand, Marke, Modell und Jahr sein?
    # ------------------------------
    
    # Die ersten 10 Autos für die Vorhersage auswählen (aus X_test)
    sample_data = X_test.head(10).copy()  # Explizite Kopie, um Warnungen zu vermeiden
    
    # Die echten Preise (y_test) zu sample_data hinzufügen, um sie zu vergleichen
    sample_data['real_price'] = y_test.head(10).values
    
    # Preise für die ersten 10 Autos vorhersagen
    predicted_prices = xgb_model.predict(sample_data.drop('real_price', axis=1))  # 'real_price' vor der Vorhersage entfernen
    
    # Die Vorhersagen zum DataFrame hinzufügen
    sample_data['predicted_price'] = predicted_prices
    
    # Nummern von 1 bis 10 für die X-Achse zuweisen
    sample_data['car_number'] = range(1, 11)
    
    # Vergleich der echten und vorhersagten Preise grafisch darstellen
    fig = px.bar(sample_data, x='car_number', y=['real_price', 'predicted_price'],
                 title="Vergleich von echten und vorhergesagten Preisen",
                 labels={"value": "Preis", "car_number": "Auto"},
                 barmode='group')
    
    # Die X-Achse anpassen, um alle Labels (1, 2, 3, ..., 10) anzuzeigen
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',  # Linearer Modus, um sicherzustellen, dass alle Labels angezeigt werden
            tick0=1,  # Der erste Wert des Labels
            dtick=1  # Abstand der Labels
        )
    )
    
    # Layout anpassen, um den Titel zu zentrieren
    fig.update_layout(
        title_x=0.5  # Titel zentrieren
    )
    
    # Grafik anzeigen
    st.plotly_chart(fig)
    
    # ------------------------------
    # Vorhersage für das Auto Mercedes-Benz A 180 Benzin 2012 122.0
    # ------------------------------
    
    # Merkmale des zu prognostizierenden Autos definieren
    car_to_predict = pd.DataFrame({
        'mileage': [67453],  # Beispiel: Kilometerstand
        'make_Mercedes-Benz': [1],  # Marke Mercedes-Benz (1 wenn vorhanden, 0 wenn nicht)
        'model_A 180': [1],  # Modell A 180
        'fuel_Gasoline': [1],  # Kraftstoff Gasoline
        'gear_automatic': [1],  # Automatikgetriebe (1 wenn vorhanden, 0 wenn nicht)
        'hp': [122],  # PS
        'year': [2012],  # Jahr
    })
    
    # Sicherstellen, dass die Spalten in car_to_predict mit denen von X_train übereinstimmen
    missing_cols = set(X.columns) - set(car_to_predict.columns)
    
    # Falls Spalten fehlen, diese hinzufügen
    car_to_predict = pd.concat([car_to_predict, pd.DataFrame({col: [0] for col in missing_cols})], axis=1)
    
    # Die Spalten in die gleiche Reihenfolge wie im Trainingsdatensatz bringen
    car_to_predict = car_to_predict[X.columns]
    
    # Merkmale des Autos normalisieren mit dem gleichen Scaler wie beim Training
    car_to_predict[["mileage", "hp", "year"]] = scaler.transform(car_to_predict[["mileage", "hp", "year"]])
    
    # Vorhersage treffen
    predicted_price = xgb_model.predict(car_to_predict)
    
    # Vorhergesagten Preis anzeigen
    st.write(f"Der vorhergesagte Preis für den Mercedes-Benz A 180 Benzin 2012 mit 122 PS beträgt: {predicted_price[0]:.2f} EUR")
    
    # ------------------------------
    # Welche Marken/Modelle behalten am besten ihren Wert?
    # ------------------------------
    
    # Wertverlust berechnen
    df["age"] = 2025 - df["year"]  # Angenommen, der Datensatz enthält Fahrzeuge bis 2025
    depreciation = df.groupby(["make", "model"])["price"].mean() / df.groupby(["make", "model"])["age"].mean()
    depreciation_df = depreciation.reset_index().rename(columns={0: "value_retention"})
    
    # Wertverlust nach Wertretention sortieren
    depreciation_df = depreciation_df.sort_values(by="value_retention", ascending=False).head(10)
    
    # Balkendiagramm erstellen
    fig2 = px.bar(depreciation_df, 
                  x="value_retention", y="model", color="make",
                  title="Modelle, die am besten ihren Wert behalten",
                  labels={"value_retention": "Wertbeibehaltung", "model": "Modell", "make": "Marke"},
                  orientation='h', 
                  category_orders={"model": depreciation_df["model"].tolist()})  # Erzwingt die Reihenfolge der Y-Achse
    
    # Layout anpassen, um den Titel zu zentrieren
    fig2.update_layout(
        title_x=0.5,  # Titel zentrieren
        yaxis={'categoryorder': 'total ascending'}  # Sicherstellen, dass nach Wert sortiert wird
    )
    
    # Grafik anzeigen
    st.plotly_chart(fig2)
    # =======================
    '''