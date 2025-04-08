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

def app(translate_text):
    
    st.header(translate_text("3. Allgemeine explorative Analyse"))
    
    # =======================
    # 2.1. Schnelles Scannen der Daten
    # =======================

    st.markdown("##### " + "**" + translate_text("3.1. Anzahl der zum Verkauf stehenden Fahrzeuge nach Zulassungsjahr") + "**")
    st.write(f"{translate_text("Die Anzahl der zum Verkauf stehenden Fahrzeuge beträgt: ")} **{len(df)}**")

    # =======================
    st.markdown("##### " + "**" + translate_text("3.2. In welchen Zeiträumen waren die Fahrzeuge zugelassen?") + "**")
    
    jahr_start, jahr_ende = df["year"].min(), df["year"].max()
    
    st.write(f"{translate_text("Die zum Verkauf stehenden Fahrzeuge haben Zulassungsjahre zwischen ")} **{jahr_start}** und **{jahr_ende}**.")

    # =======================
    
    st.markdown("##### " + "**" + translate_text("3.3. Anzahl der zum Verkauf stehenden Fahrzeuge nach Zulassungsjahr") + "**")
    
    st.write(translate_text("Die Zahl der zum Verkauf stehenden Fahrzeuge ist in den Zulassungsjahren nach 2011 höher. Die Fahrzeuge mit der höchsten Verfügbarkeit entsprechen den Zulassungsjahren 2020, 2016 und 2013. Die durchschnittliche Anzahl der Fahrzeuge pro Zulassungsjahr beträgt 4219."))

    # Anzahl der verkauften Autos pro Jahr zählen und in ein DataFrame umwandeln
    autos_pro_jahr = df["year"].value_counts().sort_index().reset_index()
    autos_pro_jahr.columns = ["jahr", "anzahl"]  # Spalten umbenennen

    # Berechne den Durchschnittswert
    avg_anzahl = autos_pro_jahr["anzahl"].mean()

    translated_year = translate_text("Jahr")
    translated_price = translate_text("Gesamtprice (€)")
    translated_amount = translate_text("Anzahl der zu verkaufenden Fahrzeuge")
    
    # Liniendiagramm mit Plotly erstellen
    fig = px.bar(autos_pro_jahr, x="jahr", y="anzahl",
              labels={"jahr": translated_year, "anzahl": translated_amount},
              width=600, height=500)

    # Horizontale Linie für den Durchschnittswert hinzufügen
    fig.add_hline(y=avg_anzahl, line_dash="dash", line_color="red",
              annotation_text=f"{translate_text("Durchschnitt: ")}{avg_anzahl:.0f}",
              annotation_position="top right")
    
    # Layout anpassen
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=translate_text("Anzahl der zu verkaufenden Fahrzeuge"),
        showlegend=False,  # Legende ausblenden
    )

    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(2011, 2022)),  # Sicherstellen, dass die Jahre 2011 bis 2021 auf der X-Achse erscheinen
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
        linecolor="black",
        range=[4000, autos_pro_jahr["anzahl"].max() + 50]
    )

    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig)
    
    # =======================

    st.markdown("##### " + "**" + translate_text("3.4. Summe der Autoverkaufspreise nach Zulassungsjahr") + "**")
    
    st.write(translate_text("Die Summe der Preise der zum Verkauf stehenden Autos steigt mit dem Jahr der Zulassung."))

    # Gesamtumsatz pro Jahr berechnen
    price_pro_jahr = df.groupby("year")["price"].sum().reset_index()

    translated_year = translate_text("Jahr")
    translated_price = translate_text("Gesamtprice (€)")
    
    # Liniendiagramm mit Plotly erstellen
    fig = px.bar(price_pro_jahr, x="year", y="price", 
                  title=translate_text("Summe der Autoverkaufspreise nach Zulassungsjahr"),
                  labels={"year": translated_year, "price": translated_price},
                  width=900, height=500)
    
    # Layout anpassen, um den Titel zu zentrieren
    fig.update_layout(
        title_x=0.5,  # Titel zentrieren
        xaxis_title=None
    )

    # Layout anpassen
    fig.update_layout(
        title_text="",
        xaxis_title=None,
        yaxis_title=translate_text("Summe der Autoverkaufspreise"),
        showlegend=False,  # Legende ausblenden
    )

    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(2011, 2022)),  # Sicherstellen, dass die Jahre 2011 bis 2021 auf der X-Achse erscheinen
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

    st.markdown("##### " + "**" + translate_text("3.5. Welche Marken sind erfasst?") + "**")
    st.write(sorted(df["make"].unique()))
    
    st.markdown("##### " + "**" + translate_text("3.6. Die Anzahl der Autos jeder Marke") + "**")
    st.write(df["make"].value_counts())

    
    # =======================
    # 3. Korrelationsanalyse (zwischen numerischen Variablen)
    # =======================
    st.subheader(translate_text("3.7. Korrelationen zwischen den (numerischen) Features"))

    st.markdown("##### " + "**" + translate_text("3.7.1. Korrelationsmatrix") + "**")
    
    st.write("- " + translate_text("Dieser Graph zeigt die Korrelation zwischen den numerischen Variablen."))
    st.write("- " + translate_text("Eine hohe positive Korrelation (rot) bedeutet, dass zwei Variablen tendenziell zusammen steigen."))
    st.write("- " + translate_text("Eine hohe negative Korrelation (blau) bedeutet, dass eine Variable steigt, während die andere fällt."))
    st.write("- " + translate_text("Steigende Leistung und sinkende Kilometerleistung stehen in engem Zusammenhang mit einem Anstieg des Fahrzeugpreises."))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.tick_params(axis='x', bottom=True, top=True, direction='inout')
    ax.tick_params(axis='y', left=True, right=True, direction='inout')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    st.pyplot(fig)
    
    st.markdown("##### " + "**" + translate_text("3.7.2. Paarweise Streudiagramme") + "**")

    st.write("- " + translate_text("Der Graph zeigt die Verteilung und Korrelationen zwischen den Features."))
    st.write("- " + translate_text("Jüngere Fahrzeuge könnten tendenziell höhere Preise aufweisen."))
    st.write("- " + translate_text("Bestimmte Muster deuten auf nicht-lineare Zusammenhänge hin."))
    
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
    
    