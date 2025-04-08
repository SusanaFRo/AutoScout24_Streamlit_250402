import streamlit as st
import page_start
import page_method
import page_explorative
import page_sales
import page_correlation
import page_price
import page_predictions
import page_recommendations
import page_conclusions

# Titel der Präsentation
st.title("Analyse des Automobilmarktes: Faktoren, die die Preisgestaltung von Fahrzeugen beeinflussen")


# Definiere die Seiten der App, die verschiedene Aspekte von Streamlit vorstellen
pages = {
    "Startseite"                                 : page_start,
    "Methodik"                                   : page_method,
    "Explorative Analyse"                        : page_explorative,
    "Verkauf und Umsatz"                         : page_sales,
    "Beziehungen zwischen Automobil-Variablen"   : page_correlation,
    "Preise "                                    : page_price,
    "Autokonfigurator: Preisvorhersage"          : page_predictions,
    "Empfehlungen nach Budget (Preisvorhersage)" : page_recommendations,
    "Schlussfolgerungen"                         : page_conclusions
}



select = st.sidebar.radio("",list(pages.keys()))

# Starte die ausgewählte Seite
pages[select].app()


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Autor:** Susana Fernández Robledo  \n**Datum:** 3. April 2025")