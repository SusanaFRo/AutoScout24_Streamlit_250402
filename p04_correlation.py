# =======================
# 1. Importieren von Bibliotheken
# =======================
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =======================
# 2. Daten laden
# =======================

@st.cache_data
def load_data():
    df = pd.read_csv("autoscout24.csv")
    return df.dropna()

df = load_data()


def app(translate_text):
       
    st.header(translate_text("4. Beziehungen zwischen Automobil-Variablen"))

    st.subheader(translate_text("4.1. Entwicklung der durchschnittlichen Fahrzeugleistung mit dem Zulassungsjahr"))

    st.write(translate_text("Die Leistung des Fahrzeugs steigt mit zunehmendem Zulassungsjahr, nimmt aber bei Zulassungen zwischen 2016 und 2018 ab."))
    
    # Sicherstellen, dass die Spalte 'year' als Integer vorliegt
    df["year"] = df["year"].astype(int)
    
    # Gruppierung: Durchschnittliche Leistung (hp) je jahr berechnen
    leistung_nach_jahr = df.groupby("year", as_index=False)["hp"].mean()
    
    # Sortierung der Daten nach jahr
    leistung_nach_jahr = leistung_nach_jahr.sort_values("year")

    translated_year = translate_text("Jahr")
    translated_hp = translate_text("Durchschnittliche Leistung (PS)")
    
    # Diagramm erstellen

    fig = px.line(
        leistung_nach_jahr,
        x="year",
        y="hp",
        #title="Entwicklung der durchschnittlichen Fahrzeugleistung über die Jahre",
        labels={
            "year": translated_year,
            "hp": translated_hp
        },
        markers=True  # Optionale Markierung der Datenpunkte
    )

    # Layout anpassen, um den Titel zu zentrieren
    fig.update_layout(title_text="")
    
    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        tickmode='array',
        title_text="",
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks (Jahre)
        tickangle=45,  
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    fig.update_yaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks (Anzahl)
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig)

    
    
    # =======================
    st.subheader(translate_text("4.2. Welche Marken und Modelle haben den niedrigsten durchschnittlichen Kilometerstand"))

    
    # Durchschnittlichen Kilometerstand pro Marke und Modell berechnen und aufsteigend sortieren
    min_mileage = df.groupby(["make", "model"])["mileage"].mean().reset_index()
    min_mileage = min_mileage.sort_values(by="mileage", ascending=True)  # Aufsteigende Reihenfolge
    
    # Die 10 Modelle mit der geringsten Laufleistung auswählen
    min_mileage_top10 = min_mileage.head(10)

    translated_hp = translate_text("Durchschnittliche Leistung (PS)")
    translated_mark = translate_text("Marke")
    translated_model = translate_text("Modell")
    translated_mileage = translate_text("Durchschnittliche Kilometerstand (km)")

    
    # Horizontales Balkendiagramm erstellen, wobei die Y-Achse korrekt sortiert ist
    fig = px.bar(min_mileage_top10, x="mileage", y="model", color="make", 
                 title=translate_text("Modelle mit der geringsten durchschnittlichen Laufleistung"),
                 labels={"mileage": translated_mileage, "model": translated_model, "make": translated_mark}, 
                 orientation='h',
                 category_orders={"model": min_mileage_top10["model"].tolist()})  
    
    # Layout anpassen, um den Titel zu zentrieren
    fig.update_layout(title_text="")
    
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
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig)


    # =======================
    st.subheader(translate_text("4.3. Wie variiert die durchschnittliche Fahrleistung je nach Brennstofftyp?"))

    st.write(translate_text("Autos mit höherer Kilometerleistung verwenden Ethanol und Diesel als Kraftstoff, während Autos mit geringerer Kilometerleistung elektrisch betrieben werden. Ein Grund für die geringere Fahrleistung von Elektroautos im Vergleich zu Autos mit Verbrennungsmotor ist, dass sie oft als Zweitwagen genutzt werden, da die Nutzer sich mehr Sorgen machen, ob die Reichweite des Elektroautos ausreicht."))
    
    # Durchschnittliche Laufleistung pro Kraftstofftyp berechnen
    avg_mileage = df.groupby("fuel")["mileage"].mean().reset_index()
    avg_mileage = avg_mileage.sort_values(by="mileage", ascending=False)  # Absteigend sortieren

    translated_fuel = translate_text("Brennstofftyp")
    translated_mileage = translate_text("Durchschnittliche Kilometerstand (km)")
    
    
    # Balkendiagramm erstellen
    fig = px.bar(avg_mileage, x="fuel", y="mileage", 
                 title=translate_text("Durchschnittliche Laufleistung nach Brennstofftyp"),
                 labels={"fuel": translated_fuel, "mileage": translated_mileage})
    
    # Layout anpassen
    fig.update_layout(title_text="")
    
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
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig)



    # =======================
    st.subheader(translate_text("4.4. Welche Art von Brennstoff ist auf dem Markt am weitesten verbreitet?"))

    st.write(translate_text("Die meisten Autos werden mit Benzin betrieben, gefolgt von Diesel und zu einem weitaus geringeren Teil mit Strom."))

    # Anzahl der Fahrzeuge pro Kraftstofftyp berechnen
    fuel_counts = df['fuel'].value_counts().reset_index()
    fuel_counts.columns = ['fuel', 'count']  # Spalten korrekt umbenennen

    translated_fuel = translate_text("Brennstofftyp")
    translated_amount = translate_text("Anzahl der Fahrzeuge")
    
    # Balkendiagramm erstellen
    fig = px.bar(fuel_counts, 
                 x='fuel', y='count', 
                 labels={'fuel': translated_fuel, 'count': translated_amount}, 
                 title="Verteilung der Brennstofftypen")
    
    # Layout anpassen
    fig.update_layout(title_text="")
    
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
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig)

    
