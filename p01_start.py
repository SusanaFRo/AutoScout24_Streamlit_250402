import streamlit as st

def app(translate_text):
    st.header(translate_text("1. Beschreibung und Ziel"))

    st.write("### " + translate_text("1.1. Beschreibung:"))
    st.write(translate_text("""
        Das Projekt „Automobilmarktanalyse: Einflussfaktoren auf die Fahrzeugpreise“ zielt darauf ab, die Einflussfaktoren auf die Fahrzeugpreise auf dem Markt zwischen 2011 und 2021 zu untersuchen. Es beinhaltet eine Datenexploration, um die Beziehungen zwischen verschiedenen Fahrzeugmerkmalen zu verstehen und sie mit ihrem Preis in Beziehung zu setzen. Mithilfe von Machine Learing wurde ein Autokonfigurator entwickelt, der den Preis eines Fahrzeugs anhand der von den Nutzern ausgewählten Merkmale vorhersagt. Darüber hinaus wurden Algorithmen angewandt, um Fahrzeuge auf der Grundlage eines bestimmten Budgets zu empfehlen.
    """))
    st.image("picture.jpg", caption=translate_text("Automobilmarktes, *AI-Bild von www.freepik.com*"))
   
  
    st.write("### " + translate_text("1.2. Ziel"))
    st.write(translate_text("""Das Hauptziel dieses Projekts ist es, den Einfluss von Faktoren auf die Fahrzeugpreisgestaltung zu analysieren und mithilfe von maschinellem Lernen eine Lösung zu schaffen, die den Nutzern hilft, das passende Fahrzeug entsprechend ihrem Budget zu konfigurieren und zu empfehlen.
    """))

