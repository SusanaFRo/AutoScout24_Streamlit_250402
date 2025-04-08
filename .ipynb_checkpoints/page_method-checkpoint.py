import streamlit as st


def app():
    st.subheader("1. Beschreibung")
    st.write("""
        Das Projekt zielt darauf ab, den Zusammenhang zwischen der Energieerzeugungsmatrix (aufgeschlüsselt nach Quellen wie fossile Brennstoffe, erneuerbare Energien, Kernenergie etc.) und der Luftqualität zu untersuchen – und wie diese beiden Faktoren die Inzidenz von schweren Krankheiten (respiratorische Erkrankungen, Herz-Kreislauf-Erkrankungen sowie Lungenkrebs) beeinflussen. Dabei werden Daten von 30 Ländern aus dem Zeitraum 2010 bis 2019 integriert, um sowohl zeitliche als auch länderspezifische Unterschiede zu analysieren.
    """)
    st.image("picture.jpg", caption="Energiewende in Europa")
   
  
    st.write("""
        ### 2. Ziel
        Das Hauptziel besteht darin, zu evaluieren, wie die Zusammensetzung der Energieerzeugung (z. B. hoher Anteil fossiler Brennstoffe gegenüber erneuerbaren oder Kernenergie) die Luftqualität beeinflusst und in der Folge die Inzidenz schwerwiegender Krankheiten beeinflusst.
        ### 3. Projektmethodik

        #### a. Datenvorverarbeitung und Bereinigung:
        - **Durchschnittswerte für alle Städte und Jahre**: Berechnung der Durchschnittswerte, um Daten nach Land und Jahr zu erhalten.
        - **Umgang mit fehlenden Werten**: Fehlende Werte werden durch den Mittelwert der jeweiligen Spalte ersetzt.

        #### b. Explorative Datenanalyse (EDA):
        - **Visualisierung von zeitlichen Trends**: Darstellung der Entwicklung der Luftverschmutzung, Erkrankungen und Energieproduktion über die Jahre hinweg.

        #### c. Statistische Modellierung und Machine Learning:

    """)
    st.write("""
    Mit Hilfe des maschinellen Lernens wurden Prognosen für die Daten erstellt, um drei Fragen zu beantworten. 
    Es wurden zwei Modelle bewertet: **LinearRegression** und **RandomForestRegressor**. 
    Das Modell mit den besten Kennzahlen (MSE, RMSE, R²) wurde für jede Prognose ausgewählt. 
    Weitere Einzelheiten zur Methodik des maschinellen Lernens finden sich im Abschnitt „Prognosen“.
    """)