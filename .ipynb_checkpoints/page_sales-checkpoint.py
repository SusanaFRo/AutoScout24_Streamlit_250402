import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten laden
file_path = "daten_ML_20250311.csv"
df = pd.read_csv(file_path, delimiter=";")

def app():

    # Numerische Spalten konvertieren (Kommas durch Punkte ersetzen und in Float umwandeln)
    cols_to_convert = [
        "PM2,5 (ug/m3)", "PM10 (ug/m3)", "NO2 (ug/m3)",
        "diseases of circulatory system (deaths per 100 000)",
        "diseases of respiratory system (deaths per 100 000)",
        "diseases of trachea/bronchus/lung cancer (deaths per 100 000)"
    ]

    for col in cols_to_convert:
        # Ensure the column is treated as a string before replacing commas
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Stileinstellungen
    sns.set_theme(style="whitegrid")



    label_mapping = {
        "PM2,5 (ug/m3)": r"PM$_{2.5}$ (µg/m³)",
        "PM10 (ug/m3)": r"PM$_{10}$ (µg/m³)",
        "NO2 (ug/m3)": r"NO$_2$ (µg/m³)",
        "diseases of circulatory system (deaths per 100 000)": "Kreislaufsystem Erkrankungen",
        "diseases of respiratory system (deaths per 100 000)": "Respiratorische Erkrankungen",
        "diseases of trachea/bronchus/lung cancer (deaths per 100 000)": "Lungenkrebs)"
    }


    # Streamlit App Title
    st.subheader("1. Zusammenhang zwischen Luftschadstoffen und Mortalitätsarten")

    st.write("""
Dieser Graph stellt mehrere lineare Regressionen zwischen den Schadstoffniveaus (PM2,5, PM10, NO₂) und den Sterblichkeitsraten verschiedener Krankheiten dar (Kreislauferkrankungen, Atemwegserkrankungen und tracheobronchial-pulmonaler Krebs).

- **PM2,5, PM10 und Sterblichkeit**: Es zeigt sich eine deutliche positive Korrelation zwischen dem Anstieg von PM2,5 und PM10 mit der Sterblichkeit bei allen analysierten Krankheiten. Die Steigung der Regressionslinie ist ausgeprägt, was auf eine starke Auswirkung hindeutet.

- **NO₂ und Sterblichkeit**: Die Beziehung scheint in einigen Fällen schwächer oder sogar inkonsistent zu sein, mit einer stärkeren Streuung der Daten. Dies deutet darauf hin, dass NO₂ möglicherweise einen weniger direkten Einfluss hat oder von anderen Faktoren beeinflusst wird.
""")


    # Variablen von Interesse
    x_vars = ["PM2,5 (ug/m3)", "PM10 (ug/m3)", "NO2 (ug/m3)"]
    y_vars = [
        "diseases of circulatory system (deaths per 100 000)",
        "diseases of respiratory system (deaths per 100 000)",
        "diseases of trachea/bronchus/lung cancer (deaths per 100 000)"
    ]

    # Figur und Achsen erstellen
    fig, axes = plt.subplots(len(y_vars), len(x_vars), figsize=(15, 12), sharex="col", sharey="row")

    # Y-Achsenbeschriftungen mit Zeilenumbrüchen
    y_labels = [
        "Diseases of circulatory system\n(deaths per 100 000)",
        "Diseases of respiratory system\n(deaths per 100 000)",
        "Diseases of trachea/bronchus/lung cancer\n(deaths per 100 000)"
    ]

    for i, y in enumerate(y_vars):
        for j, x in enumerate(x_vars):
            sns.regplot(ax=axes[i, j], data=df, x=x, y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
            
            # Y-Achsenbeschriftungen nur für die erste Spalte
            if j == 0:
                axes[i, j].set_ylabel(y_labels[i], fontsize=12)
            else:
                axes[i, j].set_ylabel("")

            # X-Achsenbeschriftungen nur für die letzte Zeile
            if i == len(y_vars) - 1:
                x_label = x.replace("NO2", "NO$_2$").replace(" (ug/m3)", " (µg/m³)")
                axes[i, j].set_xlabel(x_label, fontsize=12)
            else:
                axes[i, j].set_xlabel("")

    # Figur Titel
    #fig.suptitle("", fontsize=16, fontweight="bold")

    # Abstand zwischen Grafiken anpassen
    fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.92)

    # Plot in Streamlit anzeigen
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("2. Zusammenhang zwischen Energieerzeugung und Luftschadstoffen")

    st.markdown("""Diese Diagramme zeigen, wie die fossile Energieerzeugung (TWh) mit den PM2,5-, PM10- und NO₂-Werten zusammenhängt:
- **PM2,5, PM10 und fossile Erzeugung**: Die Beziehung ist nahezu flach, was darauf hinweist, dass die fossile Energieerzeugung möglicherweise nicht der einzige bestimmende Faktor für die PM2,5- und PM10-Werte ist.
- **NO₂ und fossile Erzeugung**: Hier ist ein leichter positiver Trend zu beobachten, was darauf hinweist, dass die fossile Energieerzeugung direkter zu den NO₂-Werten beiträgt.
""")
    
    # Numerische Spalten konvertieren (Kommas durch Punkte ersetzen und in Float umwandeln)
    cols_to_convert = [
        "PM2,5 (ug/m3)", "PM10 (ug/m3)", "NO2 (ug/m3)",
        "diseases of circulatory system (deaths per 100 000)",
        "diseases of respiratory system (deaths per 100 000)",
        "diseases of trachea/bronchus/lung cancer (deaths per 100 000)",
        "Fossil Energy Generation TWh"
    ]

    # Ensure columns are treated as strings, replace commas, and convert to float
    for col in cols_to_convert:
        df[col] = df[col].astype(str).str.replace(",", ".")  # Replace commas with dots
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, coercing errors to NaN

    # Stileinstellungen
    sns.set_theme(style="whitegrid")

    # Variablen von Interesse
    x_vars = ["Fossil Energy Generation TWh"]
    y_vars = ["PM2,5 (ug/m3)", "PM10 (ug/m3)", "NO2 (ug/m3)"]

    # Figur und Achsen erstellen
    fig, axes = plt.subplots(len(y_vars), len(x_vars), figsize=(12, 10), sharex="col", sharey="row")

    # Reshape axes if there's only one row or column
    if len(y_vars) == 1:
        axes = axes.reshape(1, -1)
    if len(x_vars) == 1:
        axes = axes.reshape(-1, 1)

    # Y-Achsenbeschriftungen mit Zeilenumbrüchen
    y_labels = [
        "PM$_{2.5}$ (µg/m³)",
        "PM$_{10}$ (µg/m³)",
        "NO$_{2}$ (µg/m³)"
    ]

    # Plotting
    for i, y in enumerate(y_vars):
        for j, x in enumerate(x_vars):
            sns.regplot(ax=axes[i, j], data=df, x=x, y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
            
            # Y-Achsenbeschriftungen nur für die erste Spalte
            axes[i, j].set_ylabel(y_labels[i], fontsize=12)

            # X-Achsenbeschriftungen nur für die letzte Zeile
            if i == len(y_vars) - 1:
                x_label = x.replace(" (ug/m3)", " (TWh)")
                axes[i, j].set_xlabel(x_label, fontsize=12)
            else:
                axes[i, j].set_xlabel("")

    # Figur Titel
    #fig.suptitle("Zusammenhang zwischen Energieerzeugung und Luftschadstoffen", fontsize=16, fontweight="bold")

    # Abstand zwischen Grafiken anpassen
    fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.92)

    # Plot in Streamlit anzeigen
    st.pyplot(fig)

    st.markdown("---")
    
    # Streamlit App Title
    st.subheader("3. Einfluss der Umweltverschmutzung auf Todesfälle durch Krankheiten zwischen 2010 und 2019 in 30 europäischen Ländern und deren Erzeugung fossiler und erneuerbarer Energie.")

    st.markdown("""Die folgende interaktive Karte wurde mit Tableau erstellt. Sie verfügt über Filter für die Konzentration von Schadstoffen, die Anzahl der Todesfälle (pro 100.000) durch umweltbedingte Krankheiten und die Jahre. Die angezeigten Informationen für jedes Land geben einen allgemeinen Überblick über alle Daten, und die Filter ermöglichen eine individuelle Anpassung der Abfragen.""")


# Tableau HTML-Code
    
    tableau_html = """
<div class='tableauPlaceholder' id='viz1742497514027' style='position: relative'>
    <noscript>
        <a href='#'>
            <img alt='Dashboard ' src='https://public.tableau.com/static/images/Um/Umweltverschmutzung_Todesfalle_EU_2010-2019/Dashboard/1_rss.png' style='border: none' />
        </a>
    </noscript>
    <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
        <param name='embed_code_version' value='3' /> 
        <param name='site_root' value='' />
        <param name='name' value='Umweltverschmutzung_Todesfalle_EU_2010-2019/Dashboard' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/Um/Umweltverschmutzung_Todesfalle_EU_2010-2019/Dashboard/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='es-ES' />
    </object>
</div>
<script type='text/javascript'>                    
    var divElement = document.getElementById('viz1742497514027');
    var vizElement = divElement.getElementsByTagName('object')[0];
    if (divElement.offsetWidth > 800) { 
        vizElement.style.minWidth='420px';
        vizElement.style.maxWidth='650px';
        vizElement.style.width='100%';
        vizElement.style.minHeight='587px';
        vizElement.style.maxHeight='887px';
        vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
    } else if (divElement.offsetWidth > 500) { 
        vizElement.style.minWidth='420px';
        vizElement.style.maxWidth='650px';
        vizElement.style.width='100%';
        vizElement.style.minHeight='587px';
        vizElement.style.maxHeight='887px';
        vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
    } else { 
        vizElement.style.width='100%';
        vizElement.style.height='727px';
    }                     
    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
"""


# Einfügen des HTML in Streamlit
    with st.container():
        st.components.v1.html(tableau_html, width=1200, height=800, scrolling=True)

