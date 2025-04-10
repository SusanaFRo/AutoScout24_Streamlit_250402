# =======================
# 1. Importieren von Bibliotheken
# =======================
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.linear_model import LinearRegression

# =======================
# 2. Daten laden
# =======================

@st.cache_data
def load_data():
    df = pd.read_csv("autoscout24.csv")
    return df.dropna()

df = load_data()


def app(translate_text):

    st.header(translate_text("5. Preise"))

    # =======================
    st.subheader(translate_text("5.1. Top 5 am häufigsten zum Kauf angebotenen Fahrzeuge und ihre Durchschnittspreise"))

    # Die fünf meistverkauften Autos ermitteln
    top_cars = df["model"].value_counts().nlargest(5).index
    filtered_df = df[df["model"].isin(top_cars)]
    
    # Durchschnittspreis der meistverkauften Autos berechnen
    top_cars_df = filtered_df.groupby(["make", "model"])["price"].mean().reset_index()
    
    # Nach dem Durchschnittspreis absteigend sortieren
    top_cars_df = top_cars_df.sort_values(by="price", ascending=False)
    
    # Plotly-Diagramm erstellen
    fig = px.bar(top_cars_df, x="model", y="price", 
                 title=translate_text("Top 5 am häufigsten zum Kauf angebotenen Fahrzeuge und ihre Durchschnittspreise"),
                 labels={"model": translate_text("Modell"), "price": translate_text("Durchschnittspreis (€)"), "make": translate_text("Marke")},
                 color="model", 
                 hover_data={"model": True, "price": True, "make": True})  # Marke im Hover anzeigen
    
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
    
    # Diagramm anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
    
     # =======================
    st.subheader(translate_text("5.2. Marke"))
    
    st.markdown("**" + translate_text("5.2.1. Durchschnittspreis pro Marke") + "**")

    st.write(translate_text("""Dieser Graph zeigt den durchschnittlichen Preis von Fahrzeugen nach Marke. Luxusmarken wie Maybach und McLaren haben deutlich höhere Durchschnittspreise als Massenmarken wie Dacia oder Chevrolet."""))

    @st.cache_data
    def compute_avg_price(df):
        return df.groupby("make")["price"].mean().reset_index().sort_values(by="price", ascending=False)
    avg_price = compute_avg_price(df)

    fig = px.bar(avg_price, x="make", y="price", 
             title=translate_text("Durchschnittlicher Preis pro Marke"),
             labels={"make": translate_text("Marke"), "price": translate_text("Durchschnittlicher Preis (€)")})

    # Layout anpassen
    fig.update_layout(title_text="")

    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("**" + translate_text("5.2.2. Preisverteilung der 10 meistverkauften Marken") + "**")

    st.write(translate_text("""Die Preisverteilung für die zehn meistverkauften Marken variiert stark. Premium-Marken wie Mercedes-Benz und BMW haben eine höhere Preisspanne als Marken wie Renault oder SEAT."""))

    top_makes = df["make"].value_counts().nlargest(10).index
    df_filtered = df[df["make"].isin(top_makes)]
    
    fig = px.violin(df_filtered, x="make", y="price", box=True, points="all",
                    title=translate_text("Preisverteilung der 10 meistverkauften Marken"),
                    hover_data=df.columns)
    
    fig.update_layout(xaxis_title=translate_text("Marke"), yaxis_title=translate_text("Preise (€)"), xaxis_tickangle=-45)

    # Layout anpassen
    fig.update_layout(title_text="")

    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
     # =======================
    st.subheader(translate_text("5.3. Power"))
    
    st.markdown("**" + translate_text("5.3.1. Welche Modelle haben das beste Preis-Leistungs-Verhältnis?") + "**")

    st.write(translate_text("""Hier wird das Preis-Leistungs-Verhältnis verschiedener Modelle dargestellt. Modelle mit einem hohen Verhältnis bieten mehr Leistung für einen niedrigeren Preis."""))


    @st.cache_data
    def compute_hp_price_ratio(df):
        df = df.copy()
        df["hp_price_ratio"] = (df["hp"] / df["price"]) * 100  # In Prozent umwandeln
        df["hp_price_ratio"] = df["hp_price_ratio"].round(1)
        ergebnis = df.groupby(["make", "model"])["hp_price_ratio"].mean().reset_index()
        return ergebnis.sort_values(by="hp_price_ratio", ascending=False)
    
    best_ratio = compute_hp_price_ratio(df)
    
    # Entfernen von doppelten Modellnamen
    best_ratio = best_ratio.drop_duplicates(subset="model")
    
    # Achse X korrekt sortieren
    best_ratio["model"] = best_ratio["model"].astype("category")
    best_ratio["model"] = best_ratio["model"].cat.set_categories(best_ratio["model"].unique(), ordered=True)
    
    # Balkendiagramm erstellen
    fig = px.bar(best_ratio, x="model", y="hp_price_ratio", 
                 title=translate_text("Bestes Leistungs-Preis-Verhältnis nach Modell"),
                 labels={"model": translate_text("Modell"), "hp_price_ratio": translate_text("Leistungs-Preis-Verhältnis (%)"), "make": translate_text("Marke")},
                 hover_data={"model": True, "hp_price_ratio": True, "make": True})  
    
    # Layout anpassen
    fig.update_layout(title_text="", xaxis_title=translate_text("Modell"), yaxis_title=translate_text("Leistungs-Preis-Verhältnis (%)"))
    
    # Achsen anpassen
    fig.update_xaxes(
        tickmode='array',
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
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
    
    #==========================
    st.markdown("**" + translate_text("5.3.2. Wie wirkt sich die Leistung (PS) auf den Preis des Fahrzeugs aus?") + "**")
    
    st.write(translate_text("""Dieser Graph zeigt die Beziehung zwischen der Leistung (PS) und dem Preis eines Fahrzeugs. Fahrzeuge mit höherer PS-Zahl tendieren dazu, teurer zu sein."""))
    
    df["year"] = pd.to_numeric(df["year"], errors='coerce')

    fig = px.scatter(df, x="hp", y="price", color="year",
                     title=translate_text("Beziehung zwischen Leistung und Preis von Autos"),
                     labels={"hp": translate_text("Leistung (PS)"), "price": translate_text("Preise (€)"), "year": translate_text("Jahr")},
                     hover_data=["make", "model", "mileage"],
                     color_continuous_scale="viridis")
    
    year_min, year_max = df["year"].min(), df["year"].max()
    tickvals = list(range(int(year_min), int(year_max) + 1, 2))
    
    # Layout anpassen
    fig.update_layout(
        title_text="",
        coloraxis_colorbar=dict(
            title=translate_text("Jahr"),
            tickvals=tickvals,
            ticktext=[str(year) for year in tickvals]
        )
    )
    
    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
     # =======================
    st.subheader(translate_text("5.4. Zulassungsjahr"))
    
    st.markdown("**" + translate_text("5.4.1. Wie wirkt sich das Zulassungsjahr auf den Preis aus?*") + "**")
    
    st.write(translate_text("""Neuere Fahrzeuge haben tendenziell höhere Preise als ältere Modelle. Der Preisverfall im Laufe der Zulassungsjahr ist deutlich erkennbar."""))
    
    fig = px.scatter(df, x="year", y="price", color="make", size="mileage", 
                 title=translate_text("Beziehung zwischen Jahr und Preis von Autos"),
                     labels={"year": translate_text("Jahr"), "price": translate_text("Preise (€)"), "make": translate_text("Marke"), "mileage": translate_text("Kilometerstand")},
                     hover_data=["model", "hp"])
    
    X_year = df["year"].values.reshape(-1, 1)
    y_price = df["price"].values
    
    model = LinearRegression()
    model.fit(X_year, y_price)
    y_pred = model.predict(X_year)
    
    fig.add_scatter(x=df["year"], y=y_pred, mode="lines", name="Trendlinie", line=dict(color="red"))
    
    fig.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=2, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    # Layout anpassen
    fig.update_layout(title_text="")

    # Anpassung der Schriftgröße für Achsentiteln und Labels
    fig.update_xaxes(
        tickmode='array',
        title_text="",
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
     # =======================
    st.subheader(translate_text("5.5. Kilometerstand"))
    
    st.markdown("**" + translate_text("5.5.1. Wie ist das Verhältnis zwischen Preis und Kilometerstand?") + "**")

    st.write(translate_text("""Fahrzeuge mit einem höheren Kilometerstand haben im Durchschnitt einen niedrigeren Preis. Dies zeigt den Wertverlust durch Nutzung."""))

    fig = px.scatter(df, x="mileage", y="price", color="year", 
                 title=translate_text("Beziehung zwischen Kilometerstand und Preis von Autos"),
                 labels={"mileage": translate_text("Kilometerstand (km)"), "price": translate_text("Preise (€)")},
                 hover_data=["make", "model", "hp"])
    
    fig.update_layout(
        title_text="",
        xaxis_title=translate_text("Kilometerstand (km)"),
        yaxis_title=translate_text("Preise (€)")
    )
    
    # Anpassung der Schriftgröße für Achsentiteln und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )

    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # =======================
    st.subheader(translate_text("5.6. Fuel"))
    
    st.markdown("**" + translate_text("5.6.1. Preisvergleich nach Brennstoffart") + "**")
    
    st.write(translate_text("""Elektrofahrzeuge und alternative Kraftstoffe zeigen eine höhere Preisspanne als herkömmliche Diesel- und Benzinfahrzeuge."""))
    
    fig = px.box(df, x="fuel", y="price", 
                 title=translate_text("Preisvergleich nach Kraftstoffart"),
                 hover_data=df.columns)
    
    fig.update_layout(xaxis_title=translate_text("Kraftstoffart"), yaxis_title=translate_text("Preise (€)"))
    
    # Layout anpassen
    fig.update_layout(title_text="")
    
    # Anpassung der Schriftgröße für Achsentiteln und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
    
    # =======================
    st.subheader(translate_text("5.7. Getriebe"))
    
    st.markdown("**" + translate_text("5.7.1. Sind Fahrzeuge mit Automatikgetriebe teurer als solche mit Schaltgetriebe?") + "**")
    
    st.write(translate_text("""Fahrzeuge mit Automatikgetriebe sind im Durchschnitt teurer als solche mit Schaltgetriebe."""))
    
    fig = px.violin(df, x="gear", y="price", box=True, 
                    title=translate_text("Preisvergleich nach Getriebetyp"),
                    labels={"gear": translate_text("Getriebe"), "price": translate_text("Preise (€)")})
    
    # Layout anpassen
    fig.update_layout(title_text="")
    
    # Anpassung der Schriftgröße für Achsentiteln und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
    
    
    # =======================
    st.subheader(translate_text("5.8. Angebotsart"))
    
    st.markdown("**" + translate_text("5.8.1. Wie groß ist der Preisunterschied zwischen den verschiedenen Arten von Angeboten?") + "**")
    
    st.write(translate_text("""Neuwagen sind erwartungsgemäß teurer als Gebrauchtwagen oder vorregistrierte Fahrzeuge."""))
    
    fig = px.box(df, x="offerType", y="price", 
                 title=translate_text("Preisverteilung nach Angebotsart"),
                 labels={"offerType": translate_text("Angebotsart"), "price": translate_text("Preise (€)")})
    
    # Layout anpassen
    fig.update_layout(title_text="")
    
    # Anpassung der Schriftgröße für Achsentiteln und Labels
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=16),  # Größe des X-Achsentitels
        tickfont=dict(size=14),  # Größe der X-Achsen-Ticks
        tickangle=45, 
        mirror=True,   
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    fig.update_yaxes(
        title_font=dict(size=16),  # Größe des Y-Achsentitels
        tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    # Diagramm in Streamlit anzeigen
    st.plotly_chart(fig, use_container_width=True)
