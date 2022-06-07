import numpy as np
import streamlit as st
import pandas as pd
from time import sleep
import torch
from src.helpers import predict_model_diagnosis, prob_model_diagnosis
from src.NNClassifiers.model_diagnosis import NNcovid_diagnosis
# import lstm
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_symptoms = torch.load('./torch_models/model_diagnosis.pth', map_location=torch.device(device))
#classifier_disease = torch.load('./torch_models/classifier_disease.pt')
#classifier_hosp = torch.load('./torch_models/classifier_hosp.pt')



#TITLE
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

    st.markdown("__Please enter patient information__")
    
    col1, col2 = st.columns(2)
    
    bool_mask = {"sí":1, "no":0, "Hombre":1, "Mujer":0}

    sexo = bool_mask[col1.radio("Sexo", ["Hombre", "Mujer"])]
    edad = col2.slider("Edad", min_value=0, max_value=100)
    
    st.markdown("__Please enter your symptoms__")
    col1, col2, col3 = st.columns(3)
    
    fiebre = bool_mask[col1.radio("Fiebre", ["No","sí"])]
    tos = bool_mask[col2.radio("Tos", ["No","sí"])]
    odinogia = bool_mask[col3.radio("Dolor al tragar", ["No","sí"])]
    
    disnea = bool_mask[col1.radio("Falta de aire", ["No","sí"])]
    irritabi = bool_mask[col2.radio("Irritabilidad", ["No","sí"])]
    diarrea = bool_mask[col3.radio("Diarrea", ["No","sí"])]
    
    dotoraci = bool_mask[col1.radio("Dolor torácico", ["No","sí"])]
    calofrios = bool_mask[col2.radio("Calofrios", ["No","sí"])]
    cefalea = bool_mask[col3.radio("Cefalea", ["No","sí"])]

    mialgias = bool_mask[col1.radio("Dolor muscular", ["No","sí"])]
    artral = bool_mask[col2.radio("Dolor de articulaciones", ["No","sí"])]
    ataedoge = bool_mask[col3.radio("Ataque al Estado General", ["No","sí"])]
    
    rinorrea = bool_mask[col1.radio("Goteo nasal", ["No","sí"])]
    polipnea = bool_mask[col2.radio("Respiración Acelerada", ["No","sí"])]
    vomito = bool_mask[col3.radio("Vomito", ["No","sí"])]
    
    dolabdo = bool_mask[col1.radio("Dolor Abdominal", ["No","sí"])]
    conjun = bool_mask[col2.radio("Conjuntivitis", ["No","sí"])]
    cianosis = bool_mask[col3.radio("Coloración azulada de la piel", ["No","sí"])]
    
    inisubis = bool_mask[col1.radio("Inicio súbito de síntomas", ["No","sí"])]
  

    st.markdown("__Please enter your commorbidities__")

    col1, col2, col3 = st.columns(3)
    
    diabetes = bool_mask[col1.radio("diabetes", ["No","sí"])]
    epoc = bool_mask[col2.radio("epoc", ["No","sí"])]
    asma = bool_mask[col3.radio("asma", ["No","sí"])]
    
    inmusupr = bool_mask[col1.radio("inmusupr", ["No","sí"])]
    hiperten = bool_mask[col2.radio("hiperten", ["No","sí"])]
    vih_sida = bool_mask[col3.radio("vih_sida", ["No","sí"])]
    
    otracon = bool_mask[col1.radio("otracon", ["No","sí"])]
    enfcardi = bool_mask[col2.radio("enfcardi", ["No","sí"])]
    obesidad = bool_mask[col3.radio("obesidad", ["No","sí"])]
    
    insrencr = bool_mask[col1.radio("insrencr", ["No","sí"])]
    tabaquis = bool_mask[col2.radio("tabaquis", ["No","sí"])]
    
    if st.button('Run classifieres'):
        st.write('Running')
        X_symptoms = np.array([[sexo, edad, fiebre, tos, odinogia, disnea, irritabi,
        			diarrea, dotoraci, calofrios, cefalea, mialgias, artral,
                    ataedoge, rinorrea, polipnea, vomito, dolabdo, conjun,
                    cianosis, inisubis]])

        res = predict_model_diagnosis(classifier_symptoms, X_symptoms, device)
        if res == 1:

            st.error("El clasificador indica que eres positivo a COVID-19")
        elif res == 0:
            st.success("El clasificador indica que eres negativo a COVID-19")
        # classfier.run(data)
        

        st.markdown("Result presentation")
        #st.snow()

with st.expander("See contributors"):
    col1, col2, col3 = st.columns(3)
    col1.markdown("[Alfonso Barajas](https://github.com/AlfonsBC)")
    col2.markdown("[Carlos Cerritos](https://github.com/carloscerlira)")
    col3.markdown("[Guillermo Cota](https://github.com/Gcota51)")
    col1.markdown("[Raul Mosqueda](https://github.com/IsaidMosqueda)")
    col2.markdown("[Artemio Padilla](https://github.com/ArtemioPadilla)")
    col3.markdown("[Pamela Ruiz](https://github.com/Pamela-ruiz9)")
