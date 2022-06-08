import numpy as np
import streamlit as st
import pandas as pd
from time import sleep
import torch
import plotly.express as px
import plotly.graph_objects as go
from src.helpers import predict_model, prob_model
from src.NNClassifiers.NNmodels import NNclassifier

#Data cases
cases_who = pd.read_csv('https://raw.githubusercontent.com/ArtemioPadilla/MachineLearningAndCOVID/main/Datasets/SDG-3-Health/WHO-COVID-19-global-data-up.csv')
aux =  cases_who[cases_who.New_cases !=0]. groupby('Country').sum()
contries = list(aux[aux.New_cases>10000].index)

# import lstm
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_symptoms = torch.load('./torch_models/model_diagnosis.pth', map_location=torch.device(device))
classifier_hosp = torch.load('./torch_models/model_hosp.pth', map_location=torch.device(device))
classifier_death = torch.load('./torch_models/model_death.pth', map_location=torch.device(device))



#TITLE
st.title("Some applications of deep learning for the COVID-19 pandemic")
st.markdown("In this application you can either get forecast for the COVID-19 pandemic infected number using **LSTMs** or you can use a **neural network classifier** to get probabilities of desease and hospitalization for an individual with certain characteristics")
st.markdown("_more info to be written here_")

analysis = st.radio("Please enter which type of application you want to explore:", ["None","Cases and deaths chart", "LSTM forecast", "Neural Network Classifier"])

if analysis == "LSTM forecast":
    with st.expander("See explanation"):
        st.markdown("EXPLANATION FOR LSTMs")
        st.code("""for i in range(5):
            algorithm""")

    country_list = ["Mexico", "USA", "Israel"]
    country = st.selectbox("Pick a country to analyse", contries)
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
    
elif analysis == "Cases and deaths chart":
    with st.expander("See explanation"):
        st.markdown("EXPLANATION FOR LSTMs")
        st.code("""for i in range(5):
            algorithm""")
    type = st.selectbox("Pick type", ["Cases", "Deaths"])
    if type == "Cases":
        fig = px.line(cases_who, x="Date_reported", y="New_cases", color="Country")
    else:
        fig = px.line(cases_who, x="Date_reported", y="New_deaths", color="Country")
    st.plotly_chart(fig)

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
    
    fiebre = bool_mask[col1.radio("Fiebre", ["no","sí"])]
    tos = bool_mask[col2.radio("Tos", ["no","sí"])]
    odinogia = bool_mask[col3.radio("Dolor al tragar", ["no","sí"])]
    
    disnea = bool_mask[col1.radio("Falta de aire", ["no","sí"])]
    irritabi = bool_mask[col2.radio("Irritabilidad", ["no","sí"])]
    diarrea = bool_mask[col3.radio("Diarrea", ["no","sí"])]
    
    dotoraci = bool_mask[col1.radio("Dolor torácico", ["no","sí"])]
    calofrios = bool_mask[col2.radio("Calofrios", ["no","sí"])]
    cefalea = bool_mask[col3.radio("Cefalea", ["no","sí"])]

    mialgias = bool_mask[col1.radio("Dolor muscular", ["no","sí"])]
    artral = bool_mask[col2.radio("Dolor de articulaciones", ["no","sí"])]
    ataedoge = bool_mask[col3.radio("Ataque al Estado General", ["no","sí"])]
    
    rinorrea = bool_mask[col1.radio("Goteo nasal", ["no","sí"])]
    polipnea = bool_mask[col2.radio("Respiración Acelerada", ["no","sí"])]
    vomito = bool_mask[col3.radio("Vomito", ["no","sí"])]
    
    dolabdo = bool_mask[col1.radio("Dolor Abdominal", ["no","sí"])]
    conjun = bool_mask[col2.radio("Conjuntivitis", ["no","sí"])]
    cianosis = bool_mask[col3.radio("Coloración azulada de la piel", ["no","sí"])]
    
    inisubis = bool_mask[col1.radio("Inicio súbito de síntomas", ["no","sí"])]
    #digcline = bool_mask[col1.radio("Neumonía", ["no","sí"])]
  

    st.markdown("__Please enter your commorbidities__")

    col1, col2, col3 = st.columns(3)
    
    diabetes = bool_mask[col1.radio("diabetes", ["no","sí"])]
    epoc = bool_mask[col2.radio("epoc", ["no","sí"])]
    asma = bool_mask[col3.radio("asma", ["no","sí"])]
    
    inmusupr = bool_mask[col1.radio("inmusupr", ["no","sí"])]
    hiperten = bool_mask[col2.radio("hiperten", ["no","sí"])]
    vih_sida = bool_mask[col3.radio("vih_sida", ["no","sí"])]
    
    otracon = bool_mask[col1.radio("otracon", ["no","sí"])]
    enfcardi = bool_mask[col2.radio("enfcardi", ["no","sí"])]
    obesidad = bool_mask[col3.radio("obesidad", ["no","sí"])]
    
    insrencr = bool_mask[col1.radio("insrencr", ["no","sí"])]
    tabaquis = bool_mask[col2.radio("tabaquis", ["no","sí"])]
    
    if st.button('Run classifieres'):

        st.markdown("""## Predicted Diagnosis""")
        X_symptoms = np.array([[sexo, edad, fiebre, tos, odinogia, disnea, irritabi,
        			diarrea, dotoraci, calofrios, cefalea, mialgias, artral,
                    ataedoge, rinorrea, polipnea, vomito, dolabdo, conjun,
                    cianosis, inisubis]])
        res = predict_model(classifier_symptoms, X_symptoms, device)
        probs = prob_model(classifier_symptoms, X_symptoms, device) 

        if res == 1:
            st.error(f"El clasificador indica que eres POSITIVO a COVID-19 una seguridad de {probs[1]*100:.2f}%")
        elif res == 0:
            if probs[0] > 0.6:
                st.success(f"El clasificador indica que eres NEGATIVO a COVID-19 una seguridad de {probs[0]*100:.2f}%")
            else:
                st.warning(f"El clasificador indica que eres NEGATIVO a COVID-19 una seguridad de {probs[0]*100:.2f}%")



        st.markdown("""## Risk of Hospitalization""")
        X_hosp= np.array([[sexo,  edad, fiebre, tos, odinogia, disnea,
                           irritabi, diarrea, dotoraci, calofrios, cefalea, mialgias,
                           artral, ataedoge, rinorrea, polipnea, vomito, dolabdo,
                           conjun, cianosis, inisubis, diabetes, epoc, asma,
                           inmusupr, hiperten, vih_sida, otracon, enfcardi, obesidad,
                           insrencr, tabaquis]])
        
        res = predict_model(classifier_hosp, X_hosp, device)
        probs = prob_model(classifier_hosp, X_hosp, device) 

        st.markdown("If you where to have covid the chances of you being hospitalized having your commobidities and symtoms are:")
        if res == 1:
            st.error(f"Estimamos que si te da COVID-19 van a tener que HOSPITALIZARTE con una seguridad de {probs[1]*100:.2f}%")
        elif res == 0:
            if probs[0] > 0.6:
                st.success(f"Estimamos que si te da COVID-19 NO van a tener que HOSPITALIZARTE con una seguridad de {probs[0]*100:.2f}%")
            else:
                st.warning(f"Estimamos que si te da COVID-19 NO van a tener que HOSPITALIZARTE con una seguridad de {probs[0]*100:.2f}%")


        st.markdown("""## Risk of Death""")

        X_death= np.array([[sexo, edad, diabetes, epoc, asma, inmusupr, hiperten,
                            vih_sida, otracon, enfcardi, obesidad, insrencr, tabaquis]])

        res = predict_model(classifier_death, X_death, device)
        probs = prob_model(classifier_death, X_death, device) 

        st.markdown("If you where to have covid given your symptoms and your commorbidities the chances of you dying are:")
        if res == 1:
            st.error(f"La red neuronal piensa que vas a MORIR con una seguridad de {probs[1]*100:.2f}%")
        elif res == 0:
            if probs[0] > 0.6:
                st.success(f"La red neuronal piensa que vas a SOBREVIVIR con una seguridad de {probs[0]*100:.2f}%")
            else:
                st.warning(f"La red neuronal piensa que vas a SOBREVIVIR con una seguridad de {probs[0]*100:.2f}%")
        # classfier.run(data)
        

        #st.snow()

with st.expander("See contributors"):
    col1, col2, col3 = st.columns(3)
    col1.markdown("[Alfonso Barajas](https://github.com/AlfonsBC)")
    col2.markdown("[Carlos Cerritos](https://github.com/carloscerlira)")
    col3.markdown("[Guillermo Cota](https://github.com/Gcota51)")
    col1.markdown("[Raul Mosqueda](https://github.com/IsaidMosqueda)")
    col2.markdown("[Artemio Padilla](https://github.com/ArtemioPadilla)")
    col3.markdown("[Pamela Ruiz](https://github.com/Pamela-ruiz9)")
