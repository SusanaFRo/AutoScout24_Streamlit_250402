import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.inspection import permutation_importance


def app(translate_text):
    st.header(translate_text("2. Projektmethodik"))

    st.write("### " + translate_text("2.1. Daten:"))
    st.write(translate_text("Für dieses Projekt wurden die Daten der zum Verkauf stehenden Autos 'autoscout24.csv' verwendet (Datenquelle von") + " [kaggle](https://www.kaggle.com/datasets/ander289386/cars-germany)), "
         "die die folgenden Spalten enthalten:")
    st.markdown(f"- **{translate_text('mileage:')}** {translate_text(' Der Kilometerstand des Fahrzeugs.')}")
    st.markdown(f"- **{translate_text('model:')}** {translate_text(' Das Modell des Fahrzeugs.')}")
    st.markdown(f"- **{translate_text('gear:')}** {translate_text(' Das Getriebe des Fahrzeugs (z.B. manuell oder automatisch).')}")
    st.markdown(f"- **{translate_text('price:')}** {translate_text(' Der Preis des Fahrzeugs in Euro.')}")
    st.markdown(f"- **{translate_text('hp:')}** {translate_text(' Die Leistung des Fahrzeugs in Pferdestärken (PS).')}")
    st.markdown(f"- **{translate_text('year:')}** {translate_text(' Jahr der ersten Zulassung des Fahrzeugs.')}")


    st.write("### " + translate_text("2.2. Datenvorverarbeitung und Bereinigung:"))
    st.write("- " + translate_text("Umgang mit fehlenden Werten: Zeilen mit null Werten wurden gelöscht."))

    st.write("### " + translate_text("2.3. Explorative Datenanalyse (EDA):"))

    st.markdown(f"- **{translate_text('Allgemeine explorative Analyse:')}** {translate_text(' Es wird eine allgemeine Untersuchung der Daten, einschließlich der Korrelationsmatrix der numerischen Merkmale und der paarweisen Streudiagramme, durchgeführt.')}")
    st.markdown(f"- **{translate_text('Beziehungen zwischen Automobil-Variablen:')}** {translate_text(' Einige Fragen zu den Beziehungen zwischen den Variablen in den Fahrzeugdaten werden beantwortet.')}")
    st.markdown(f"- **{translate_text('Preise:')}** {translate_text(' Es wird der Einfluss der verschiedenen Fahrzeugvariablen auf den Verkaufspreis des Fahrzeugs dargestellt. Enthält auch die 5 am häufigsten zum Kauf angebotenen Fahrzeuge.')}")

    
    st.write("### " + translate_text("2.4. Statistische Modellierung und Machine Learning:"))
    
    st.write(translate_text("Mit Hilfe von Machine Learning wurden die auf den folgenden Seiten dargestellten Ergebnisse erzielt:"))
    
    st.markdown(f"- **{translate_text('Autokonfigurator: Preisvorhersage:')}** {translate_text(' Auf dieser Seite ist es möglich, die Daten eines Autos einzugeben und den voraussichtlichen Preis mit Hilfe eines Machine Learning-Modells zu ermitteln.')}")
    st.markdown(f"- **{translate_text('Empfehlungen nach Budget:')}** {translate_text(' Auf dieser Seite können Sie einen Kostenvoranschlag eingeben, um die Liste der Fahrzeuge mit dem gewünschten Preis zu erhalten. Sie können auch auswählen, ob Sie reale Preise (die direkt aus den Daten gewonnen werden) oder simulierte Preise (die mithilfe des Machine Learning-Modells gewonnen werden) verwenden möchten.')}")

    st.write("#### " + translate_text("2.4.1. Auswahl des Modells:"))

    st.write(translate_text("Es wurden drei Modelle bewertet: LinearRegression, RandomForestRegressor und XGBoost:"))
    st.markdown(f"1. **{translate_text('LinearRegression Modell:')}** {translate_text(' Geht von einer linearen Beziehung zwischen den Eingangsvariablen (X) und der Zielvariablen (Y) aus. Wird verwendet, wenn wenige Daten vorhanden sind und Interpretierbarkeit gewünscht wird.')}")
    st.markdown(f"2. **{translate_text('RandomForestRegressor Modell:')}** {translate_text(' Erzeugt mehrere Bäume und mittelt deren Vorhersagen, was die Genauigkeit verbessert und Überanpassung reduziert. Wird verwendet, wenn die Beziehung zwischen den Variablen nicht linear oder komplex ist, Interaktionen zwischen Variablen bestehen und genauere Vorhersagen benötigt werden (reduziert die Varianz).')}")
    st.markdown(f"3. **{translate_text('XGBoost:')}** {translate_text(' XGBoost (Extreme Gradient Boosting) ist ein Ensemble-Algorithmus, der hauptsächlich für Klassifizierungs- und Regressionsaufgaben verwendet wird. Er basiert auf dem Konzept des Boosting, bei dem mehrere schwache Modelle (in diesem Fall Entscheidungsbäume) kombiniert werden, um ein starkes Modell zu bilden. Er ermöglicht die Verarbeitung nichtlinearer Beziehungen und bietet eine bessere Leistung und Fähigkeit zur Verarbeitung großer und komplexer Daten als Random Forest.')}")
    st.write(translate_text("Zur Bewertung der Modelle wurden die Metriken (MSE, RMSE, R²) verglichen. Im Allgemeinen hat ein Modell eine bessere Leistung bei niedrigen MSE-, RMSE-Werten und einem R² nahe 1. RMSE gibt den Fehler an, den das Modell im Durchschnitt macht, und R² zeigt, wie gut das Modell die Variabilität der Daten erklärt. Die Ergebnisse der Metriken sind in der nachstehenden Tabelle aufgeführt. Die besten Metriken wurden für das Modell XGBoots erzielt, weshalb es in diesem Projekt verwendet wird."))


    data = {
        "MODELL": ["LinearRegression", "RandomForestRegressor", "XGBoost"],
        "MSE": [2916.25, 1915.02, 2200.06],
        "RMSE": [6162.71, 6815.00, 6120.29],
        "R2": [0.89, 0.86, 0.89]
    }
    
    df = pd.DataFrame(data)
    
    # Zu markierende Zeilen 
    highlight_indices = [2]
    
    # Funktion zur Umwandlung von DataFrame in HTML mit schwarzem Text und Farben
    def dataframe_to_html(df, highlight_indices):
        html = """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f4f4f4;
                color: black;  /* Color negro para el texto del encabezado */
            }
            .highlight {
                background-color: rgba(255, 255, 150, 0.5);  /* Amarillo claro con transparencia */
                color: black;  /* Texto negro */
            }
        </style>
        <table>
        """
        
        # Überschriften
        html += "<tr>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
        
        # Filas
        for i, row in df.iterrows():
                row_class = "highlight" if i in highlight_indices else ""
                html += f"<tr class='{row_class}'>" + "".join(
                [f"<td>{row[col]:.2f}</td>" if isinstance(row[col], float) else f"<td>{row[col]}</td>" for col in df.columns]
            ) + "</tr>"
        
        html += "</table>"
        return html
    
    # DataFrame in HTML umwandeln und in Streamlit anzeigen
    html_table = dataframe_to_html(df, highlight_indices)
    st.markdown(html_table, unsafe_allow_html=True)



    st.write("#### " + translate_text("2.4.2. Analyse der XGBoost-Autopreisprognose"))
    
    @st.cache_data
    def load_data():
        df = pd.read_csv("autoscout24.csv").dropna()
        return df.sample(n=5000, random_state=42)
    
    df = load_data()
    
    features = ["mileage", "make", "model", "fuel", "gear", "hp", "year"]
    X = pd.get_dummies(df[features], columns=["make", "model", "fuel", "gear"], drop_first=True)
    y = df["price"]
    
    scaler = StandardScaler()
    X[["mileage", "hp", "year"]] = scaler.fit_transform(X[["mileage", "hp", "year"]])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    @st.cache_resource
    def train_model():
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    model = train_model()
    
    y_pred = model.predict(X_test)
    
    st.write("##### " + translate_text("A. Permutationswichtigkeit (XGBoost)"))
    st.write(translate_text("- Dieser Graph zeigt die Wichtigkeit der einzelnen Variablen für das XGBoost-Modell."))
    st.write(translate_text("- Die Variable „hp“ (Horsepower) hat den größten Einfluss auf die Vorhersagen."))
    st.write(translate_text("- Marken und Modellnamen sind ebenfalls wichtige Faktoren."))

    
    @st.cache_data
    def feature_importance():
        importance = pd.Series(model.feature_importances_, index=X_train.columns).nlargest(10)
        fig = px.bar(importance, title=None)
        fig.update_layout(xaxis_title=translate_text("Features"), yaxis_title=translate_text("Importance"))
        return fig
    
    st.plotly_chart(feature_importance(), use_container_width=True)
    
    st.write("##### " + translate_text("B. Vergleich tatsächlicher Preis vs. Vorhergesagter Preis"))

    st.write(translate_text("- Der Graph zeigt die Übereinstimmung zwischen den tatsächlichen und den vorhergesagten Preisen."))
    st.write(translate_text("- Eine perfekte Vorhersage würde dazu führen, dass alle Punkte auf der Diagonalen liegen."))
    st.write(translate_text("- Leichte Abweichungen sind sichtbar, aber das Modell scheint eine gute Annäherung zu liefern."))
    
    @st.cache_data
    def price_comparison():
        
        translated_predicted_price = translate_text("Vorhergesagter Preis (€)")
        translated_actual_price = translate_text("Tatsächlicher Preis (€)")
        
        df_compare = pd.DataFrame({translated_predicted_price: y_test, translated_actual_price: y_pred})
    
        fig = px.scatter(df_compare, x=translated_actual_price, y=translated_predicted_price, trendline="ols")

        # Layout anpassen
        fig.update_layout(title_text="")
    
        # Anpassung der Schriftgröße für Achsentitel und Labels
        fig.update_xaxes(
            title=translated_actual_price,
            tickmode='array',
            title_font=dict(size=16),  # Größe des X-Achsentitels
            tickfont=dict(size=14),  # Größe der X-Achsen-Ticks (Jahre)
            tickangle=45,  
            mirror=True,   
            showline=True,
            linewidth=1,
            linecolor="black"
        )
    
        fig.update_yaxes(
            title=translated_predicted_price,
            title_font=dict(size=16),  # Größe des Y-Achsentitels
            tickfont=dict(size=14),  # Größe der Y-Achsen-Ticks (Anzahl)
            mirror=True,
            showline=True,
            linewidth=1,
            linecolor="black"
        )
        
        return fig
    
    st.plotly_chart(price_comparison(), use_container_width=True)
    
    
    
    st.write("##### " + translate_text("C. Verteilung des Residualfehlers"))
    
    st.write(translate_text("- Die Histogramm zeigt die Fehlerverteilung der Vorhersagen."))
    st.write(translate_text("- Ein symmetrischer Fehler um Null deutet darauf hin, dass das Modell keine systematische Verzerrung aufweist."))
    st.write(translate_text("- Größere Fehler in beiden Richtungen könnten auf nicht erfasste Muster oder Ausreißer hinweisen."))
    
    
    def residual_analysis():
        
        translated_error = translate_text("Residual fehler")
        translated_freq = translate_text("Häufigkeit")
    
        df_residuals = pd.DataFrame({translated_error: y_test - y_pred})
    
        fig = px.histogram(df_residuals, x=translated_error, nbins=50, labels={translated_error: translated_error})
    
        # Layout anpassen
        fig.update_layout(title_text="")
    
        # Anpassung der Schriftgröße für Achsentitel und Labels
        fig.update_xaxes(
            title=translated_error,
            title_font=dict(size=16),
            tickfont=dict(size=14),
            tickangle=45,
            mirror=True,
            showline=True,
            linewidth=1,
            linecolor="black"
        )
    
        fig.update_yaxes(
            title=translated_freq,
            title_font=dict(size=16),
            tickfont=dict(size=14),
            mirror=True,
            showline=True,
            linewidth=1,
            linecolor="black"
        )
    
        return fig

    st.plotly_chart(residual_analysis(), use_container_width=True)

    
    # ==============================

    st.write("##### " + translate_text("""2.4.3. Vergleich der tatsächlichen und prognostizierten Preise für die 5 am häufigsten zum Kauf angebotenen Fahrzeuge"""))

    # Die fünf am häufigsten zum Kauf angebotenen Fahrzeuge ermitteln
    top_cars = df["model"].value_counts().nlargest(5).index
    filtered_df = df[df["model"].isin(top_cars)].copy()
    
    # Durchschnittspreis der am häufigsten zum Kauf angebotenen Fahrzeuge berechnen
    filtered_df = filtered_df.groupby(["make", "model"], as_index=False).agg({"price": "mean"})
    
    # Umwandlung der kategorialen Variablen
    top_cars_df = pd.get_dummies(filtered_df, columns=["make", "model"], drop_first=True)
    
    # Fehlende Spalten in top_cars_df ergänzen
    missing_cols = set(X_train.columns) - set(top_cars_df.columns)
    for col in missing_cols:
        top_cars_df[col] = 0
    
    # Spalten neu sortieren
    top_cars_df = top_cars_df[X_train.columns]
    
    # Vorhersagen treffen
    predicted_prices = model.predict(top_cars_df)
    
    # Vorhergesagte Preise hinzufügen
    filtered_df["predicted_price"] = predicted_prices
    filtered_df["predicted_price"] = filtered_df["predicted_price"].round(2)
    filtered_df["price"] = filtered_df["price"].round(2)
    
    # Übersetzung von Beschriftungen und Hover
    translated_mark = translate_text("Mark")
    translated_model = translate_text("Modell")
    translated_price = translate_text("Preise (€)")
    translated_predicted_price = translate_text("Vorhergesagter Preis (€)")
    
    filtered_df = filtered_df.rename(columns={
        "make": translated_mark,
        "model": translated_model,
        "price": translated_price,
        "predicted_price": translated_predicted_price
    })
    
    # Vergleich plotten
    fig = px.bar(
        filtered_df, x=translated_model, y=[translated_price, translated_predicted_price],
        barmode="group",
        labels={translated_model: translated_model, translated_mark: translated_mark, translated_price: translated_price, translated_predicted_price: translated_predicted_price}
    )
    
    # Tooltips anpassen
    fig.update_traces(
        hovertemplate=f'{translated_model}: <b>%{{x}}</b><br>{translated_mark}: %{{customdata[0]}}<br>{translated_price}: %{{y:.2f}} €',
        customdata=filtered_df[[translated_mark]].values
    )
    
    # Layout anpassen
    fig.update_layout(title_text="")
    
    # Anpassung der Schriftgröße für Achsentitel und Labels
    fig.update_xaxes(
        title=translated_model,
        title_font=dict(size=16),
        tickfont=dict(size=14),
        tickangle=45,
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    fig.update_yaxes(
        title=translated_price,
        title_font=dict(size=16),
        tickfont=dict(size=14),
        mirror=True,
        showline=True,
        linewidth=1,
        linecolor="black"
    )
    
    st.plotly_chart(fig, use_container_width=True)
