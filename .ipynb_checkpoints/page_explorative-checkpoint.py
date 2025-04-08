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
# 2. Laden und Vorverarbeitung der Daten
# =======================

df = pd.read_csv('autoscout24.csv', sep=',', decimal='.')


def app():
    
    st.header("2. Explorative Analyse")
    
    # =======================
    # 2.1. Schnelles Scannen der Daten
    # =======================

    st.subheader("2.1. Wieviele Autos wurden verkauft?")
    
    st.write("""
        **Luftqualitätsmessungen**
        - **PM2.5**: Feinstaub mit einem Durchmesser von 2,5 Mikrometern oder weniger
        - **PM10**: Feinstaub mit einem Durchmesser von 10 Mikrometern oder weniger
        - **NO2**: Stickstoffdioxid
    """)

    n_autos_verkauft = df["year"].count()
    st.write(f"Die Zahl der verkauften Autos beträgt: **{n_autos_verkauft}**")
    
    st.subheader("2.2. Über welchen Zeitraum?")
    jahr_start = df["year"].min()
    jahr_ende = df["year"].max()
    st.write(f"Die Verkäufe erfolgten zwischen **{jahr_start}** und **{jahr_ende}**.")
    
    st.subheader("2.3. Welche Marken sind erfasst?")
    mark_sort = sorted(df["make"].unique())
    st.write(mark_sort)
    
    st.subheader("2.4. Die Anzahl der Autos jeder Marke")
    count_mark = df["make"].value_counts()
    st.write(count_mark)
    
    # =======================
    # 3. Korrelationsanalyse (zwischen numerischen Variablen)
    # =======================
    st.subheader("2.5. Existieren Korrelationen zwischen den (numerischen) Features?")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.subheader("2.6. Paarweise Streudiagramme")
    pairplot_fig = sns.pairplot(df, hue="year")
    st.pyplot(pairplot_fig)
    
