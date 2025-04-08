import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.inspection import permutation_importance


def app():
    st.header("2. Projektmethodik")

    st.write("""### 2.1. Daten:""")

    
    st.write("Für dieses Projekt wurden die Daten „autoscout24.csv“ verwendet, die die folgenden Spalten enthalten:")

    st.write("""
    - **mileage**: "Autokilometerstand - Der Kilometerstand des Fahrzeugs.
    - **model**: "Automodell - Das Modell des Fahrzeugs.",
    "fuel": "Fahrzeugkraftstoff - Der Kraftstofftyp des Fahrzeugs (z.B. Benzin, Diesel).",
    - **gear**: "Fahrzeuggetriebe - Das Getriebe des Fahrzeugs (z.B. manuell oder automatisch).",
    - **price**: "Preis des Fahrzeugs - Der Preis des Fahrzeugs in Euro.",
    - **hp**: "Fahrzeugleistung in PS - Die Leistung des Fahrzeugs in Pferdestärken (PS).",
    - **year**: "Jahr - Jahr der ersten Zulassung des Fahrzeugs."
    """)
    
    st.write("""
### 2.2. Datenvorverarbeitung und Bereinigung:
- **Umgang mit fehlenden Werten**: Zeilen mit null Werten wurden gelöscht.

### 2.3. Explorative Datenanalyse (EDA):
- **Allgemeine explorative Analyse**: Es wird eine allgemeine Untersuchung der Daten, einschließlich der Korrelationsmatrix der numerischen Merkmale und der paarweisen Streudiagramme, durchgeführt.
- **Beziehungen zwischen Automobil-Variablen**: Einige Fragen zu den Beziehungen zwischen den Variablen in den Fahrzeugdaten werden beantwortet.
- **Preise**: DEs wird der Einfluss der verschiedenen Fahrzeugvariablen auf den Verkaufspreis des Fahrzeugs dargestellt.

### 2.4. Statistische Modellierung und Machine Learning:
Mit Hilfe von Machine Learning wurden die auf den folgenden Seiten dargestellten Ergebnisse erzielt:
- **Autokonfigurator: Preisvorhersage**: Auf dieser Seite ist es möglich, die Daten eines Autos einzugeben und den voraussichtlichen Preis mit Hilfe eines Machine Learning-Modells zu ermitteln.
- **Empfehlungen nach Budget**: Auf dieser Seite können Sie einen Kostenvoranschlag eingeben, um die Liste der Fahrzeuge mit dem gewünschten Preis zu erhalten. Sie können auch auswählen, ob Sie reale Preise (die direkt aus den Daten gewonnen werden) oder simulierte Preise (die mithilfe des Machine Learning-Modells gewonnen werden) verwenden möchten.

#### 2.4.1. Auswahl des Modells:
Es wurden drei Modelle bewertet: **LinearRegression**, **RandomForestRegressor** und **XGBoost**:
1. **LinearRegression Modell**: Geht von einer linearen Beziehung zwischen den Eingangsvariablen (X) und der Zielvariablen (Y) aus. Wird verwendet, wenn wenige Daten vorhanden sind und Interpretierbarkeit gewünscht wird.
2. **RandomForestRegressor Modell**: Erzeugt mehrere Bäume und mittelt deren Vorhersagen, was die Genauigkeit verbessert und Überanpassung reduziert. Wird verwendet, wenn die Beziehung zwischen den Variablen nicht linear oder komplex ist, Interaktionen zwischen Variablen bestehen und genauere Vorhersagen benötigt werden (reduziert die Varianz).
3. **XGBoost**: XGBoost (Extreme Gradient Boosting) ist ein Ensemble-Algorithmus, der hauptsächlich für Klassifizierungs- und Regressionsaufgaben verwendet wird. Er basiert auf dem Konzept des Boosting, bei dem mehrere schwache Modelle (in diesem Fall Entscheidungsbäume) kombiniert werden, um ein starkes Modell zu bilden. Er ermöglicht die Verarbeitung nichtlinearer Beziehungen und bietet eine bessere Leistung und Fähigkeit zur Verarbeitung großer und komplexer Daten als Random Forest.

Zur Bewertung der Modelle wurden die Metriken (MSE, RMSE, R²) verglichen. Im Allgemeinen hat ein Modell eine bessere Leistung bei niedrigen MSE-, RMSE-Werten und einem R² nahe 1. RMSE gibt den Fehler an, den das Modell im Durchschnitt macht, und R² zeigt, wie gut das Modell die Variabilität der Daten erklärt.""")


    data = {
        "MODELL": ["LinearRegression", "RandomForestRegressor", "XGBoost"],
        "MSE": [2916.25, 1915.02, 2200.06],
        "RMSE": [6162.71, 6815.00, 6120.29],
        "R2": [0.89, 0.86, 0.89]
    }
    
    df = pd.DataFrame(data)
    
    # Zu markierende Zeilen 
    highlight_indices = [2]
    
    # Funktion zur Umwandlung von DataFrame in HTML mit schwarzem Text und Farben
    def dataframe_to_html(df, highlight_indices):
        html = """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f4f4f4;
                color: black;  /* Color negro para el texto del encabezado */
            }
            .highlight {
                background-color: rgba(255, 255, 150, 0.5);  /* Amarillo claro con transparencia */
                color: black;  /* Texto negro */
            }
        </style>
        <table>
        """
        
        # Überschriften
        html += "<tr>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
        
        # Filas
        for i, row in df.iterrows():
                row_class = "highlight" if i in highlight_indices else ""
                html += f"<tr class='{row_class}'>" + "".join(
                [f"<td>{row[col]:.2f}</td>" if isinstance(row[col], float) else f"<td>{row[col]}</td>" for col in df.columns]
            ) + "</tr>"
        
        html += "</table>"
        return html
    
    # DataFrame in HTML umwandeln und in Streamlit anzeigen
    html_table = dataframe_to_html(df, highlight_indices)
    st.markdown(html_table, unsafe_allow_html=True)


    st.write("""#### 2.4.2. Analyse der XGBoost-Autopreisprognose""")

    # Daten laden und vorbereiten
    @st.cache_data
    def load_data():
        df = pd.read_csv("autoscout24.csv").dropna()
        return df.sample(n=5000, random_state=42)  # Subset für bessere Performance
    
    df = load_data()
    
    # Merkmale und Zielvariable definieren
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = pd.get_dummies(df[features], columns=["make", "model", "fuel", "gear"], drop_first=True)
    y = df["price"]
    
    # Daten normalisieren
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modell trainieren
    @st.cache_resource
    def train_model():
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    model = train_model()
    
    y_pred = model.predict(X_test)

    st.write("""##### A. Permutationswichtigkeit (XGBoost)""")
    st.write(""" 
    - Dieser Graph zeigt die Wichtigkeit der einzelnen Variablen für das XGBoost-Modell.
    - Die Variable „hp“ (Horsepower) hat den größten Einfluss auf die Vorhersagen.
    - Marken und Modellnamen sind ebenfalls wichtige Faktoren.""")
    
    # Merkmalswichtigkeit
    @st.cache_data
    def feature_importance():
        fig = px.bar(pd.Series(model.feature_importances_, index=X_train.columns).nlargest(10))
    
        # Anpassung der Schriftgröße für Achsentitel und Labels
        fig.update_xaxes(
            tickmode='array',
            title_font=dict(size=16),  # Größe des X-Achsentitels
            tickfont=dict(size=14),  # Größe der X-Achsen-Ticks (Jahre)
            tickangle=45,  
            mirror=True,   
            showline=True,
            linewidth=1,
            linecolor="black"
        )
    
        fig.update_yaxes(
            title_font=dict(size=16),  # Größe des Y-Achsentitels
            tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks (Anzahl)
            mirror=True,
            showline=True,
            linewidth=1,
            linecolor="black"
        )
        return fig
        
    st.plotly_chart(feature_importance(), use_container_width=True)

    st.write("""##### B. Vergleich tatsächlicher Preis vs. Vorhergesagter Preis""")

    st.write(""" 
    - Der Graph zeigt die Übereinstimmung zwischen den tatsächlichen und den vorhergesagten Preisen.
    - Eine perfekte Vorhersage würde dazu führen, dass alle Punkte auf der Diagonalen liegen.
    - Leichte Abweichungen sind sichtbar, aber das Modell scheint eine gute Annäherung zu liefern.""")
    
    # Vorhersagevergleich
    @st.cache_data
    def price_comparison():
        fig = px.scatter(pd.DataFrame({"Tatsächlicher Preis": y_test, "Vorhergesagter Preis": y_pred}), x="Tatsächlicher Preis", y="Vorhergesagter Preis", trendline="ols")
        # Anpassung der Schriftgröße für Achsentitel und Labels
        fig.update_xaxes(
            tickmode='array',
            title_font=dict(size=16),  # Größe des X-Achsentitels
            tickfont=dict(size=14),  # Größe der X-Achsen-Ticks (Jahre)
            tickangle=45,  
            mirror=True,   
            showline=True,
            linewidth=1,
            linecolor="black"
        )
    
        fig.update_yaxes(
            title_font=dict(size=16),  # Größe des Y-Achsentitels
            tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks (Anzahl)
            mirror=True,
            showline=True,
            linewidth=1,
            linecolor="black"
        )
        return fig
    
    st.plotly_chart(price_comparison(), use_container_width=True)

    st.write("""##### C. Verteilung des Residualfehlers""")

    st.write(""" 
    - Die Histogramm zeigt die Fehlerverteilung der Vorhersagen.
    - Ein symmetrischer Fehler um Null deutet darauf hin, dass das Modell keine systematische Verzerrung aufweist.
    - Größere Fehler in beiden Richtungen könnten auf nicht erfasste Muster oder Ausreißer hinweisen.""")
    
    # Residualanalyse
    @st.cache_data
    def residual_analysis():
        fig = px.histogram(pd.DataFrame({"Residualfehler": y_test - y_pred}), x="Residualfehler", nbins=50)

        # Anpassung der Schriftgröße für Achsentitel und Labels
        fig.update_xaxes(
            tickmode='array',
            title_font=dict(size=16),  # Größe des X-Achsentitels
            tickfont=dict(size=14),  # Größe der X-Achsen-Ticks (Jahre)
            tickangle=45,  
            mirror=True,   
            showline=True,
            linewidth=1,
            linecolor="black"
        )
    
        fig.update_yaxes(
            title_font=dict(size=16),  # Größe des Y-Achsentitels
            tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks (Anzahl)
            mirror=True,
            showline=True,
            linewidth=1,
            linecolor="black"
        )
        return fig
    
    st.plotly_chart(residual_analysis(), use_container_width=True)



    # ------------------------------
    # 📌 Vergleich der tatsächlichen und prognostizierten Preise für die 5 meistverkauften Autos
    # ------------------------------

    st.write("""#### 2.4.3. Vergleich der tatsächlichen und prognostizierten Preise für die 5 meistverkauften Autos""")

    # Die fünf meistverkauften Autos ermitteln
    top_cars = df["model"].value_counts().nlargest(5).index
    filtered_df = df[df["model"].isin(top_cars)].copy()
    
    # Durchschnittspreis der meistverkauften Autos berechnen
    filtered_df = filtered_df.groupby(["make", "model"], as_index=False).agg({"price": "mean"})
    
    # Umwandlung der kategorialen Variablen
    top_cars_df = pd.get_dummies(filtered_df, columns=["make", "model"], drop_first=True)
    
    # Fehlende Spalten in top_cars_df ergänzen
    missing_cols = set(X_train.columns) - set(top_cars_df.columns)
    for col in missing_cols:
        top_cars_df[col] = 0
    
    # Spalten neu sortieren
    top_cars_df = top_cars_df[X_train.columns]
    
    # Vorhersagen treffen
    predicted_prices = model.predict(top_cars_df)
    
    # Vorhergesagte Preise hinzufügen
    filtered_df["predicted_price"] = predicted_prices
    filtered_df["predicted_price"] = filtered_df["predicted_price"].round(2)
    filtered_df["price"] = filtered_df["price"].round(2)
    
    # Vergleich plotten
    fig = px.bar(
        filtered_df, x="model", y=["price", "predicted_price"],
        title="Vergleich von Realen und Vorhergesagten Preisen",
        labels={"value": "Preis (in €)", "model": "Auto"},
        barmode="group"
        #text_auto='.2f'
    )
    
    # Tooltips anpassen
    fig.update_traces(
        hovertemplate='Modell: <b>%{x}</b><br>Marke: %{customdata[0]}<br>Preis: %{y:.2f} €',
        customdata=filtered_df[["make"]].values
    )

    # Layout anpassen
    fig.update_layout(title_text="")
    
    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        title_font=dict(size=16),
        tickfont=dict(size=14),
        tickangle=45,
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    fig.update_yaxes(
        title_font=dict(size=16),
        tickfont=dict(size=14),
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    
    