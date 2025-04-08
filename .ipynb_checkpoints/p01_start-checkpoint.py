import streamlit as st


def app():
    st.subheader("1. Beschreibung")
    st.write("""
        Das Projekt "Analyse des Automobilmarktes: Faktoren, die die Preisgestaltung von Fahrzeugen beeinflussen" zielt darauf ab, die Faktoren zu untersuchen, die die Preisgestaltung von Fahrzeugen auf dem Markt zwischen 2011 und 2021 beeinflussen. Es beinhaltet eine umfassende Datenexploration, um die Beziehungen zwischen verschiedenen Variablen wie Fahrzeugmerkmalen, Produktionskosten und Marktbedingungen zu verstehen. Unter Verwendung von maschinellem Lernen wird ein Autokonfigurator entwickelt, der es Nutzern ermöglicht, ein Fahrzeug entsprechend ihrem Budget auszuwählen. Zudem werden Empfehlungsalgorithmen implementiert, die Fahrzeuge basierend auf einem angegebenen Budget empfehlen.
    """)
    st.image("picture.jpg", caption="Automobilmarktes, *AI-Bild von www.freepik.com*")
   
  
    st.write("""
        ### 2. Ziel
        Das Hauptziel dieses Projekts ist es, den Einfluss von Faktoren auf die Fahrzeugpreisgestaltung zu analysieren und mithilfe von maschinellem Lernen eine Lösung zu schaffen, die den Nutzern hilft, das passende Fahrzeug entsprechend ihrem Budget zu konfigurieren und zu empfehlen.
    """) 