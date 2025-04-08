import streamlit as st
from deep_translator import GoogleTranslator
import p01_start
import p02_method
import p03_explorative
import p04_correlation
import p05_price
import p06_predictions
import p07_recommendations
import p08_conclusions

# Definir idiomas disponibles
LANGUAGES = {
    "de": "Deutsch",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "it": "Italiano"
}

# Sprachauswahl in der Seitenleiste
selected_lang = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])

# Funktion der Übersetzung
def translate_text(text):
    if selected_lang == "de":  # Wenn die Sprache Deutsch ist, nicht übersetzen
        return text
    return GoogleTranslator(source="auto", target=selected_lang).translate(text)

# Übergabe der Sprache an die Seiten
pages = {
    translate_text("Startseite"): p01_start,
    translate_text("Methodik"): p02_method,
    translate_text("Allgemeine explorative Analyse"): p03_explorative,
    translate_text("Beziehungen zwischen Automobil-Variablen"): p04_correlation,
    translate_text("Preise"): p05_price,
    translate_text("Autokonfigurator: Preisvorhersage"): p06_predictions,
    translate_text("Empfehlungen nach Budget"): p07_recommendations,
    translate_text("Schlussfolgerungen"): p08_conclusions,
}

# Übersetzten Titel anzeigen
st.title(translate_text("Analyse des Automobilmarktes: Faktoren, die die Preisgestaltung von Fahrzeugen beeinflussen"))

# Übersetzte Seite auswählen
select = st.sidebar.radio("", list(pages.keys()))

# Starten der gewählten Seite und Umschalten auf die Übersetzungsfunktion.
pages[select].app(translate_text=translate_text)

# Footer übersetzt
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{translate_text('Autor')}:** Susana Fernández Robledo  \n**{translate_text('Datum')}:** 3. April 2025")
