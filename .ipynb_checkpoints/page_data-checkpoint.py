import streamlit as st
import pandas as pd

df = pd.read_csv("daten_AK.csv",sep=";", dtype={'Jahr': str})

def app():
    st.header("Datequellen")
    st.write("""
        - **Luftqualität**: PM2.5, PM10, NO2 Messungen
             - Datenquelle: [WHO](https://www.who.int/data/gho/data/themes/air-pollution/who-air-quality-database/2022)
        - **Gesundheit**: Chronische respiratorische Erkrankungen, Kreislaufsystemerkrankungen, Lungenkrebs
             - Datenquelle: [WHO](https://gateway.euro.who.int/en/datasets/european-health-for-all-database/)
        - **Energieproduktion**: Fossile Energie, Kernenergie, Erneuerbare Energie
             - Datenquelle: [Kaggle](https://www.kaggle.com/datasets/mexwell/renewable-vs-fossil-in-energy-in-europe)
    """)
            
    st.subheader("Die Daten von 30 europäischen Ländern für den Zeitraum 2010-2019")
    st.write("""
        **Luftqualitätsmessungen**
        - **PM2.5**: Feinstaub mit einem Durchmesser von 2,5 Mikrometern oder weniger
        - **PM10**: Feinstaub mit einem Durchmesser von 10 Mikrometern oder weniger
        - **NO2**: Stickstoffdioxid
    """)

    df_air = df.rename(columns={
    "ISO3": "Land",
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "Jahr": "Jahr"
    })[["Land", "Jahr", "PM2.5", "PM10", "NO2"]]
    st.write(df_air)

    st.write("""
        **Todesfälle pro 100.000 Einwohner**
        - **Kreislaufsystem**
        - **Respiratorische**
        - **Lungenkrebs**
    """)
    df_disease = df.rename(columns={
    "ISO3": "Land",
    "KE": "Kreislaufsystem Erkrankungen",
    "RE": "Respiratorische Erkrankungen",
    "LE": "Lungenkrebs",
    "Jahr": "Jahr"
    })[["Land", "Jahr","Kreislaufsystem Erkrankungen", "Respiratorische Erkrankungen", "Lungenkrebs"]]
    st.write(df_disease)

    st.write("""
        **Eneregieproduktion in TWh**
        - **Fossile Energie**: Kohle, Gas
        - **Kern**
        - **Erneuerbar**: Wind, Solar, Hydro, Bio
    """)

    df_energy = df.rename(columns={
    "ISO3": "Land",
    "FEn": "Fossile Energie",
    "KEn": "Kernenergie",
    "EEn": "Erneuerbare Energie",
    "EKEn": "Erneuerbare + Kernenergie",
    "Jahr": "Jahr"
    })[["Land", "Jahr","Fossile Energie", "Kernenergie", "Erneuerbare Energie","Erneuerbare + Kernenergie"]]
    st.write(df_energy)

    st.subheader("Abkürzungen")
    st.write("""
        AUT	Austria\n 
        BEL Belgium\n 
        BGR	Bulgaria\n 
        HRV	Croatia\n 
        CYP	Cyprus\n 
        CZE	Czechia\n 
        DNK	Denmark\n 
        FIN	Finland\n 
        FRA	France\n 
        DEU	Germany\n 
        GRC	Greece\n 
        HUN	Hungary\n 
        ISL	Iceland\n 
        IRL	Ireland\n 
        ITA	Italy\n 
        LVA	Latvia\n 
        LTU	Lithuania\n 
        LUX	Luxembourg\n 
        MLT	Malta\n 
        NLD	Netherlands\n 
        NOR	Norway\n 
        POL	Poland\n 
        PRT	Portugal\n 
        ROU	Romania\n 
        SVK	Slovakia\n 
        SVN	Slovenia\n 
        ESP	Spain\n 
        SWE	Sweden\n 
        CHE	Switzerland\n 
        GBR	United Kingdom

    """)