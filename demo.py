import streamlit as st
import pandas as pd
from time import sleep
# import lstm
# import classifier

st.title("Some applications of deep learning for the COVID-19 pandemic")


st.markdown("In this application you can either get forecast for the COVID-19 pandemic infected number using **LSTMs** or you can use a **neural network classifier** to get probabilities of desease and hospitalization for an individual with certain characteristics")

st.markdown("_more info to be written here_")

analysis = st.radio("Please enter which type of application you want to explore:", ["None","LSTM forecast", "Neural Network Classifier"])

if analysis == "LSTM forecast":
    with st.expander("See explanation"):
        st.markdown("EXPLANATION FOR LSTMs")
        st.code("""for i in range(5):
            algorithm""")

    country_list = ["Mexico", "USA", "Israel"]
    country = st.selectbox("Pick a country to analyse", country_list)
    st.slider(label= "Select the window of time in days to predict", min_value=1, max_value=30, value=None, step=None)
    window_to_predict = st.write(f"The country you choose is {country}")
    st.metric(label = "Test", value=1, delta=-0.1, delta_color="normal")
    # Example plot
    # Get LSTM prediction for country
    # data, prediction = lstm(country, window_to_predict)
    # Plot line chart
    # st.line_chart(np.concat([data,prediction]))
    url = "https://raw.githubusercontent.com/ArtemioPadilla/ML-Datasets/main/Casos_Diarios_Estado_Nacional_Defunciones_20210121.csv"
    df = pd.read_csv(url)
    #st.dataframe(df.iloc[:,3:])
    st.line_chart(df.iloc[:,3:])
elif analysis == "Neural Network Classifier":
    with st.expander("See explanation"):
        st.markdown("EXPLANATION FOR CLASSIFIER")
        st.code("""for i in range(5):
            algorithm""")

    st.markdown("Please enter patient information")
    col1, col2 = st.columns(2)
    sex = col1.radio("Sexo", ["Hombre", "Mujer"])
    edad = col2.slider("Edad", min_value=0, max_value=100)
    
    st.markdown("Please enter your symptoms")
    col1, col2, col3 = st.columns(3)
    
    fiebre = col1.radio("Fiebre", ["sí", "No"])
    tos = col2.radio("Tos", ["sí", "No"])
    odinogia = col3.radio("Dolor al tragar", ["sí", "No"])
    
    disnea= col1.radio("Falta de aire", ["sí", "No"])
    diarrea = col2.radio("Diarrea", ["sí", "No"])
    dotoraci = col3.radio("Dolor torácico", ["sí", "No"])
    
    calofrios= col1.radio("Calofrios", ["sí", "No"])
    cefalea = col2.radio("Cefalea", ["sí", "No"])
    mialgias = col3.radio("Dolor muscular", ["sí", "No"])
    
    artral= col1.radio("Dolor de articulaciones", ["sí", "No"])
    ataedoge = col2.radio("Ataque al Estado General", ["sí", "No"])
    rinorrea = col3.radio("Goteo nasal", ["sí", "No"])
    
    polipnea= col1.radio("Respiración Acelerada", ["sí", "No"])
    vomito = col2.radio("Vomito", ["sí", "No"])
    dolabdo = col3.radio("Dolor Abdominal", ["sí", "No"])

    
    conjun= col1.radio("Conjuntivitis", ["sí", "No"])
    cianosis = col2.radio("Coloración azulada de la piel", ["sí", "No"])
    inisubis = col3.radio("Inicio súbito de síntomas", ["sí", "No"])
  

    st.markdown("Please enter your commorbidities")

    col1, col2, col3 = st.columns(3)
    
    diabetes = col1.radio("diabetes", ["sí", "No"])
    epoc = col2.radio("epoc", ["sí", "No"])
    asma = col3.radio("com3", ["sí", "No"])
    
    inmusupr = col1.radio("com4", ["sí", "No"])
    hiperten = col2.radio("com5", ["sí", "No"])
    vih_sida = col3.radio("com6", ["sí", "No"])
    
    inmusupr = col1.radio("inmusupr", ["sí", "No"])
    hiperten = col2.radio("hiperten", ["sí", "No"])
    vih_sida = col3.radio("vih_sida", ["sí", "No"])
    
    otracon = col1.radio("otracon", ["sí", "No"])
    enfcardi = col2.radio("enfcardi", ["sí", "No"])
    obesidad = col3.radio("obesidad", ["sí", "No"])
    
    insrencr = col1.radio("insrencr", ["sí", "No"])
    tabaquis = col2.radio("tabaquis", ["sí", "No"])
    if st.button('Run classifier'):
        st.write('Running')
        # classfier.run(data)
        sleep(0.5)
        st.success("Done")

        st.markdown("Result presentation")
        st.snow()

with st.expander("See contributors"):
    col1, col2, col3 = st.columns(3)
    col1.markdown("[Alfonso Barajas](https://github.com/AlfonsBC)")
    col2.markdown("[Carlos Cerritos](https://github.com/carloscerlira)")
    col3.markdown("[Guillermo Cota](https://github.com/Gcota51)")
    col1.markdown("[Raul Mosqueda](https://github.com/IsaidMosqueda)")
    col2.markdown("[Artemio Padilla](https://github.com/ArtemioPadilla)")
    col3.markdown("[Pamela Ruiz](https://github.com/Pamela-ruiz9)")
