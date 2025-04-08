import streamlit as st

def app(translate_text):


    st.header(translate_text("8. Schlussfolgerungen und Verkaufsempfehlungen"))
    

    st.subheader(translate_text("8.1. Schlussfolgerungen"))

    st.markdown(f"- {translate_text('Die Anzahl der zum Verkauf stehenden Fahrzeuge beträgt 46405 und sie haben Zulassungsjahre zwischen 2011 und 2021. Die durchschnittliche Anzahl der Fahrzeuge pro Zulassungsjahr beträgt 4219. Der Gesamtpreis steigt mit dem Jahr der Registrierung stark an.')}")
    
    st.markdown(f"- **{translate_text('Fahrzeuge mit mehr Leistung')}** {translate_text(' erzielen in der Regel auch höhere Preise.')}")
    
    st.markdown(f"- **{translate_text('Fahrzeuge mit Benzin- und Dieselmotoren dominieren weiterhin den Markt,')}** {translate_text(' während Elektrofahrzeuge eine geringere durchschnittliche Kilometerleistung aufweisen. Dies deutet darauf hin, dass Elektroautos oft als Zweitfahrzeuge genutzt werden.')}")
    
    st.markdown(f"- **{translate_text('Neuere Fahrzeuge erzielen höhere Verkaufspreise,')}** {translate_text(' wobei der Wertverlust mit zunehmendem Alter erheblich ist. Ein höherer')} **{translate_text(' Kilometerstand')}** {translate_text(' führt generell zu niedrigeren Verkaufspreisen, allerdings gibt es Ausnahmen je nach Modell und Marke.')}")
    
    st.markdown(f"- **{translate_text('Neuere Fahrzeuge erzielen höhere Verkaufspreise.')}** {translate_text(' Fahrzeuge mit')} **{translate_text('Automatikgetriebe')}** {translate_text(' sind tendenziell teurer als solche mit Schaltgetriebe. Neuwagen und vorregistrierte Fahrzeuge erzielen die höchsten Preise, während Gebrauchtwagen und Mitarbeiterfahrzeuge günstiger sind.')}")
    
    st.markdown(f"- {translate_text('Der Volkswagen Golf, Ford Focus und Fiesta sowie der Opel Astra und Corsa sind die am häufigsten zum Verkauf stehenden Modelle, was darauf hindeutet, dass sie sowohl beim Kauf als auch beim Wiederverkauf beliebte Optionen sind.')}")
    
    st.markdown(f"- {translate_text('Luxusmarken wie Maybach, McLaren und Aston Martin weisen die höchsten Durchschnittspreise auf.')}")
    
    st.markdown(f"- {translate_text('Die Implementierung von ')} **{translate_text('Preisvorhersagemodellen')}** {translate_text(' kann helfen, die optimalen Verkaufszeitpunkte für Fahrzeuge zu bestimmen.')}")
    

    
    st.subheader(translate_text("8.2. Verkaufsempfehlungen"))

    st.markdown(f"- **{translate_text('Optimierung des Fahrzeugangebots:')}** {translate_text('Der Fokus auf Benzin- und Dieselfahrzeuge bleibt lukrativ, aber Elektrofahrzeuge sollten aufgrund wachsender Nachfrage ebenfalls verstärkt angeboten werden. Fahrzeuge mit geringem Kilometerstand und Zulassungsjahr ab 2018 erzielen höhere Preise und sollten bevorzugt vermarktet werden.')}")
    
    st.markdown(f"- **{translate_text('Preisstrategien anpassen:')}** {translate_text('Hochleistungsfahrzeuge und Luxusmarken sollten gezielt beworben werden, da sie höhere Gewinnmargen bieten. Modelle mit gutem Preis-Leistungs-Verhältnis, wie der Mercedes-Benz CLC oder Renault Grand Espace, sollten als attraktive Optionen für preisbewusste Käufer positioniert werden.')}")
    
    st.markdown(f"- **{translate_text('Angebotsart strategisch nutzen:')}** {translate_text('Vorregistrierte Fahrzeuge und Demonstrationswagen können als attraktive Alternativen zu Neuwagen vermarktet werden, da sie oft hohe Rabatte bieten, aber nahezu neuwertig sind. Gebrauchtwagen mit niedriger Laufleistung sollten mit Garantieoptionen beworben werden, um das Vertrauen der Käufer zu stärken.')}")
    
    st.markdown(f"- **{translate_text('Gezielte Verkaufsförderung:')}** {translate_text('Die Nachfrage nach Automatikfahrzeugen steigt, Händler sollten daher eine größere Auswahl in diesem Segment bereitstellen. Saisonale Angebote oder Finanzierungsmodelle für Elektrofahrzeuge könnten helfen, den Absatz in diesem Bereich zu steigern.')}")
    
    st.markdown(f"- **{translate_text('Digitale Vertriebswege stärken:')}** {translate_text('Online-Plattformen und digitale Showrooms sollten intensiver genutzt werden, um eine größere Käuferschicht zu erreichen.')}")
    
