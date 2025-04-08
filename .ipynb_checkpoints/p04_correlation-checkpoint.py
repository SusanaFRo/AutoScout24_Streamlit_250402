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

@st.cache_resource
def app():
       
    st.header("4. Beziehungen zwischen Automobil-Variablen")

    st.subheader("4.1. Entwicklung der durchschnittlichen Fahrzeugleistung über die Jahre")

    st.write("""Die Leistung der Autos hat zwischen 2016 und 2018 stark zugenommen.""")
    
    # Sicherstellen, dass die Spalte 'year' als Integer vorliegt
    df["year"] = df["year"].astype(int)
    
    # Gruppierung: Durchschnittliche Leistung (hp) je jahr berechnen
    leistung_nach_jahr = df.groupby("year", as_index=False)["hp"].mean()
    
    # Sortierung der Daten nach jahr
    leistung_nach_jahr = leistung_nach_jahr.sort_values("year")
    
    # Diagramm erstellen

    fig = px.line(
        leistung_nach_jahr,
        x="year",
        y="hp",
        #title="Entwicklung der durchschnittlichen Fahrzeugleistung über die Jahre",
        labels={
            "year": "Jahr",
            "hp": "Durchschnittliche Leistung (PS)"
        },
        markers=True  # Optionale Markierung der Datenpunkte
    )

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
    st.subheader("4.2. Welche Marken und Modelle haben den niedrigsten durchschnittlichen Kilometerstand")

    
    # Durchschnittlichen Kilometerstand pro Marke und Modell berechnen und aufsteigend sortieren
    min_mileage = df.groupby(["make", "model"])["mileage"].mean().reset_index()
    min_mileage = min_mileage.sort_values(by="mileage", ascending=True)  # Aufsteigende Reihenfolge
    
    # Die 10 Modelle mit der geringsten Laufleistung auswählen
    min_mileage_top10 = min_mileage.head(10)
    
    # Horizontales Balkendiagramm erstellen, wobei die Y-Achse korrekt sortiert ist
    fig = px.bar(min_mileage_top10, x="mileage", y="model", color="make", 
                 title="Modelle mit der geringsten durchschnittlichen Laufleistung",
                 labels={"mileage": "Durchschnittliche Laufleistung (PS)", "model": "Modell", "make": "Marke"}, 
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
    st.subheader("4.3. Wie variiert die durchschnittliche Fahrleistung je nach Brennstofftyp?")

    st.write("""Autos mit höherer Kilometerleistung verwenden Ethanol und Diesel als Kraftstoff, während Autos mit geringerer Kilometerleistung elektrisch betrieben werden. Ein Grund für die geringere Fahrleistung von Elektroautos im Vergleich zu Autos mit Verbrennungsmotor ist, dass sie oft als Zweitwagen genutzt werden, da die Nutzer sich mehr Sorgen machen, ob die Reichweite des Elektroautos ausreicht.""")
    
    # Durchschnittliche Laufleistung pro Kraftstofftyp berechnen
    avg_mileage = df.groupby("fuel")["mileage"].mean().reset_index()
    avg_mileage = avg_mileage.sort_values(by="mileage", ascending=False)  # Absteigend sortieren
    
    # Balkendiagramm erstellen
    fig = px.bar(avg_mileage, x="fuel", y="mileage", 
                 title="Durchschnittliche Laufleistung nach Brennstofftyp",
                 labels={"fuel": "Brennstofftyp", "mileage": "Durchschnittliche Laufleistung"})
    
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
    st.subheader("4.4. Welche Art von Brennstoff ist auf dem Markt am weitesten verbreitet?")

    st.write("""Die meisten Autos werden mit Benzin betrieben, gefolgt von Diesel und zu einem weitaus geringeren Teil mit Strom.""")

    # Anzahl der Fahrzeuge pro Kraftstofftyp berechnen
    fuel_counts = df['fuel'].value_counts().reset_index()
    fuel_counts.columns = ['fuel', 'count']  # Spalten korrekt umbenennen
    
    # Balkendiagramm erstellen
    fig = px.bar(fuel_counts, 
                 x='fuel', y='count', 
                 labels={'fuel': 'Brennstofftyp', 'count': 'Anzahl der Fahrzeuge'}, 
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

    
