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
    st.header("🚗 6. Autokonfigurator: Preisvorhersage")
    
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
        f"Jahr (Min: {min_year}, Max: {max_year})",
        min_value=min_year, 
        max_value=max_year_limit, 
        value=max_year, 
        step=1,  # Nur ganze Zahlen
        help=f"Min: {min_year}, Max: {max_year_limit}"
    )
    
    # Bedingung, um den DataFrame zu erstellen und die Vorhersage nur auszuführen, wenn der Button gedrückt wird
    if st.button("🔍 Preis vorhersagen"):
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
        st.success(f"💰 Der geschätzte Preis beträgt: {predicted_price_str} EUR")

        st.write("""*Die Ergebnisse basieren auf Preisvorhersagen, die mit Hilfe von Machine Learning erstellt wurden.*""")

    '''
    # =======================
    

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