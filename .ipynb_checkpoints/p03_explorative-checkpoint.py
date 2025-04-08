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
# 1. Daten laden mit Cache
# =======================
@st.cache_data
def load_data():
    return pd.read_csv('autoscout24.csv', sep=',', decimal='.')

df = load_data()

@st.cache_resource
def app():
    st.header("3. Allgemeine explorative Analyse")
    
    # =======================
    # 2.1. Schnelles Scannen der Daten
    # =======================
    st.subheader("3.1. Wieviele Autos wurden verkauft?")
    st.write(f"Die Zahl der verkauften Autos beträgt: **{len(df)}**")
    
    st.subheader("3.2. In welchen Zeiträumen waren die Fahrzeuge zugelassen?")
    jahr_start, jahr_ende = df["year"].min(), df["year"].max()
    st.write(f"Die Verkäufe erfolgten zwischen **{jahr_start}** und **{jahr_ende}**.")
    
    st.subheader("3.3. Welche Marken sind erfasst?")
    st.write(sorted(df["make"].unique()))
    
    st.subheader("3.4. Die Anzahl der Autos jeder Marke")
    st.write(df["make"].value_counts())
    
    # =======================
    # 3. Korrelationsanalyse (zwischen numerischen Variablen)
    # =======================
    st.subheader("3.5. Existieren Korrelationen zwischen den (numerischen) Features?")
    
    st.write("""
    - Dieser Graph zeigt die Korrelation zwischen den numerischen Variablen.
    - Eine hohe positive Korrelation (rot) bedeutet, dass zwei Variablen tendenziell zusammen steigen.
    - Eine hohe negative Korrelation (blau) bedeutet, dass eine Variable steigt, während die andere fällt.
    - Steigende Leistung und sinkende Kilometerleistung stehen in engem Zusammenhang mit einem Anstieg des Fahrzeugpreises. """)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.tick_params(axis='x', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', left=True, right=True, direction='inout')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    st.pyplot(fig)
    
    st.subheader("3.6. Paarweise Streudiagramme")
    
    st.write("""
    - Der Graph zeigt die Verteilung und Korrelationen zwischen den Features.
    - Jüngere Fahrzeuge könnten tendenziell höhere Preise aufweisen.
    - Bestimmte Muster deuten auf nicht-lineare Zusammenhänge hin.""")
    
    sample_df = df.sample(min(500, len(df)))  # Reduzierung der Datenmenge für schnellere Visualisierung
    pairplot_fig = sns.pairplot(sample_df, hue="year")
    
    for ax_row in pairplot_fig.axes:
        for ax in ax_row:
            if ax is not None:
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                ax.tick_params(axis='x', rotation=45, bottom=True, top=True, direction='inout')
                ax.tick_params(axis='y', left=True, right=True, direction='inout')
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
    
    st.pyplot(pairplot_fig)
    
    