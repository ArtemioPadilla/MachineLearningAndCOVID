import numpy as np
import streamlit as st
import pandas as pd
import pickle
from time import sleep
import torch
import plotly.express as px
import plotly.graph_objects as go
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from src.helpers import predict_model, prob_model
from src.NNClassifiers.NNmodels import NNclassifier
from src.Forecast.LSTM_ARIMA import predict_future_jojojo,plot_ts, LSTM


#Data cases
cases_who = pd.read_csv('https://raw.githubusercontent.com/ArtemioPadilla/MachineLearningAndCOVID/main/Datasets/SDG-3-Health/WHO-COVID-19-global-data-up.csv')

# import lstm
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache
def load_model(path):
	  return torch.load(path,  map_location=torch.device(device))

classifier_symptoms = load_model('./torch_models/model_diagnosis.pth')
classifier_hosp = load_model('./torch_models/model_hosp.pth')
classifier_death = load_model('./torch_models/model_death.pth')

#afganistan = load_model('./torch_models/LSTMS_models_New_cases/Afghanistan_New_cases.pth')

#afganistan2 = load_model('./torch_models/LSTMS_models_New_deaths/ArgentinaNew_deaths.pth')

#classifier_symptoms = torch.load('./torch_models/model_diagnosis.pth', map_location=torch.device(device))
#classifier_hosp = torch.load('./torch_models/model_hosp.pth', map_location=torch.device(device))
#classifier_death = torch.load('./torch_models/model_death.pth', map_location=torch.device(device))



#TITLE
st.title("Some applications of deep learning for the COVID-19 pandemic")
st.markdown("In this application you can either get forecast for the COVID-19 pandemic infected number and deaths number using **LSTMs & ARIMA** or you can use a **neural network classifier** to get probabilities of desease and hospitalization for an individual with certain characteristics")


analysis = st.radio("Please enter which type of application you want to explore:", ["About us","Cases and deaths chart", "LSTM-ARIMA forecast", "Convolutional Neural Networks"])

if analysis == "LSTM-ARIMA forecast":
    with st.expander("See explanation"):
        st.markdown("What is RNN LSTM?")
        st.markdown("LSTM cells are used in recurrent neural networks that learn to predict the future from sequences of variable lengths. That RNN work with any kind of sequential data and, unlike ARIMA are not restricted to time series. ")
        st.markdown("LSTM architecture consists of a set of recurrently connected subnets. The basic architecture of a LSTM RNN is:")
        st.image("https://raw.githubusercontent.com/ArtemioPadilla/MachineLearningAndCOVID/main/src/Forecast/RNN_LSTM.JPG")  
        st.markdown("What is ARIMA?")
        st.markdown("ARIMA is a class of time series prediction models, and the name is an abbreviation for AutoRegressive Integrated Moving Average. The backbone of ARIMA is a mathematical model that represents the time series values using its past values. The model is based  in past values and past errors.")
        st.markdown("The main objective of this section is to perform a comparison of predictions using an lstm neural network and arima time series model.")
        st.markdown("To compare the models, the following metrics were calculed for the countries: Kenya, South Africa, China, India, France, Georgia, Mexico, United States of America")
        st.markdown(" - RMSE")
        st.markdown(" - MSE")
        st.markdown(" - MAE")


    type_ts = st.radio("Count Type", ["New cases","New deaths"])

    if type_ts == 'New cases':
        with st.expander("Metrics"):
            st.markdown("Countries: Kenya, South Africa, China, India, France, Georgia, Mexico, United States of America")
            st.code("""    ------------LSTM Kenya---------------
    El MSE con LSTM es:  212143.2
    El RMSE con LSTM es:  460.59006
    El MAE con LSTM es:  180.14561

    ------------ARIMA Kenya---------------
    El MSE con ARIMA es:  740323.7511088527
    El RMSE con ARIMA es:  860.4206826366116
    El MAE con ARIMA es:  390.5434566216013


    ------------LSTM South Africa---------------
    El MSE con LSTM es:  9446242.0
    El RMSE con LSTM es:  3073.4739
    El MAE con LSTM es:  1612.4082

    ------------ARIMA South Africa---------------
    El MSE con ARIMA es:  171186698.77508742
    El RMSE con ARIMA es:  13083.833489275512
    El MAE con ARIMA es:  12446.598950552743


    ------------LSTM China---------------
    El MSE con LSTM es:  437632700.0
    El RMSE con LSTM es:  20919.672
    El MAE con LSTM es:  10398.583

    ------------ARIMA China---------------
    El MSE con ARIMA es:  1066199612.6180129
    El RMSE con ARIMA es:  32652.712178592654
    El MAE con ARIMA es:  18538.37500169928


    ------------LSTM India---------------
    El MSE con LSTM es:  115993200.0
    El RMSE con LSTM es:  10770.014
    El MAE con LSTM es:  5018.1313

    ------------ARIMA India---------------
    El MSE con ARIMA es:  9428567365.85269
    El RMSE con ARIMA es:  97100.81032541742
    El MAE con ARIMA es:  46162.13585655936


    ------------LSTM France---------------
    El MSE con LSTM es:  10029152000.0
    El RMSE con LSTM es:  100145.66
    El MAE con LSTM es:  63065.78

    ------------ARIMA France---------------
    El MSE con ARIMA es:  16628049406.33834
    El RMSE con ARIMA es:  128949.79413065514
    El MAE con ARIMA es:  82924.88815696917


    ------------LSTM Georgia---------------
    El MSE con LSTM es:  22785006.0
    El RMSE con LSTM es:  4773.3643
    El MAE con LSTM es:  2275.1643

    ------------ARIMA Georgia---------------
    El MSE con ARIMA es:  43724340.12655043
    El RMSE con ARIMA es:  6612.438289054229
    El MAE con ARIMA es:  4475.968398733317


    ------------LSTM Mexico---------------
    El MSE con LSTM es:  122454904.0
    El RMSE con LSTM es:  11065.935
    El MAE con LSTM es:  4777.158

    ------------ARIMA Mexico---------------
    El MSE con ARIMA es:  397645536.81367713
    El RMSE con ARIMA es:  19941.051547340154
    El MAE con ARIMA es:  9434.531578083092


    ------------LSTM United States of America---------------
    El MSE con LSTM es:  20236818000.0
    El RMSE con LSTM es:  142256.17
    El MAE con LSTM es:  74350.38

    ------------ARIMA United States of America---------------
    El MSE con ARIMA es:  68172416934.0904
    El RMSE con ARIMA es:  261098.4812941094
    El MAE con ARIMA es:  155405.6977868798""")

    elif type_ts == "New deaths":
        with st.expander("Metrics"):
            st.markdown("Countries: Kenya, South Africa, China, India, France, Georgia, Mexico, United States of America")
            st.code("""    ------------LSTM Kenya---------------
    El MSE con LSTM es:  9.267876
    El RMSE con LSTM es:  3.0443187
    El MAE con LSTM es:  1.8983194

    ------------ARIMA Kenya---------------
    El MSE con ARIMA es:  12.484331549287267
    El RMSE con ARIMA es:  3.5333173575674275
    El MAE con ARIMA es:  2.1373411751674865


    ------------LSTM South Africa---------------
    El MSE con LSTM es:  4450.0024
    El RMSE con LSTM es:  66.708336
    El MAE con LSTM es:  38.15697

    ------------ARIMA South Africa---------------
    El MSE con ARIMA es:  8342.607891533537
    El RMSE con ARIMA es:  91.33787763865294
    El MAE con ARIMA es:  52.80795956130448


    ------------LSTM China---------------
    El MSE con LSTM es:  592.30865
    El RMSE con LSTM es:  24.337393
    El MAE con LSTM es:  15.833975

    ------------ARIMA China---------------
    El MSE con ARIMA es:  11965.511929528428
    El RMSE con ARIMA es:  109.38698245005402
    El MAE con ARIMA es:  67.43896391264572


    ------------LSTM India---------------
    El MSE con LSTM es:  117735.234
    El RMSE con LSTM es:  343.12567
    El MAE con LSTM es:  119.17878

    ------------ARIMA India---------------
    El MSE con ARIMA es:  193645.3129197657
    El RMSE con ARIMA es:  440.05148894165296
    El MAE con ARIMA es:  306.6602972624162


    ------------LSTM France---------------
    El MSE con LSTM es:  7795.0635
    El RMSE con LSTM es:  88.28966
    El MAE con LSTM es:  64.197334

    ------------ARIMA France---------------
    El MSE con ARIMA es:  10335.722447465152
    El RMSE con ARIMA es:  101.6647551881435
    El MAE con ARIMA es:  74.53796328080226


    ------------LSTM Georgia---------------
    El MSE con LSTM es:  52.66859
    El RMSE con LSTM es:  7.257313
    El MAE con LSTM es:  5.0376315

    ------------ARIMA Georgia---------------
    El MSE con ARIMA es:  1680.009230823292
    El RMSE con ARIMA es:  40.987915668197765
    El MAE con ARIMA es:  35.83540573395437


    ------------LSTM Mexico---------------
    El MSE con LSTM es:  393.85153
    El RMSE con LSTM es:  19.845694
    El MAE con LSTM es:  11.482425

    ------------ARIMA Mexico---------------
    El MSE con ARIMA es:  27996.721057525814
    El RMSE con ARIMA es:  167.32220730532399
    El MAE con ARIMA es:  138.43139017245866


    ------------LSTM United States of America---------------
    El MSE con LSTM es:  1088132.8
    El RMSE con LSTM es:  1043.136
    El MAE con LSTM es:  645.5833

    ------------ARIMA United States of America---------------
    El MSE con ARIMA es:  1023231.7525496285
    El RMSE con ARIMA es:  1011.5491844441517
    El MAE con ARIMA es:  798.7112836557212""")


    type_ts_ = type_ts.replace(" ", "_")
    aux =  cases_who[cases_who[type_ts_] !=0]. groupby('Country').sum()
    countries = list(aux[aux[type_ts_]>10000].index)
    country = st.selectbox("Pick a country to analyse", countries)
    window_to_predict = st.slider(label= "Select the window of time in days to predict", min_value=1, max_value=30, value=None, step=None)
    st.write(f"The country you choose is {country}")
    
    #st.metric(label = "Test", value=1, delta=-0.1, delta_color="normal")
    # Example plot
    # Get LSTM prediction for country
    # data, prediction = lstm(country, window_to_predict)
    # Plot line chart
    # st.line_chart(np.concat([data,prediction]))
    window = 4
    sc = MinMaxScaler()
    type_ts_ = type_ts.replace(" ", "_")
    #trainX,trainY= data_train(country, window, window_to_predict,type_ts_ )

    #Entrenamos
    #lstm = load_model('./torch_models/LSTMS_models_New_cases/Afghanistan_New_cases.pth')
    #lstm_save = load_model('./torch_models/LSTM_models_'+str(type_ts_)+'/'+country+'_New_cases.pth')
    if type_ts_=="New_cases":
        last_pth = '_New_cases.pth'
    else:
        last_pth = 'New_deaths.pth'

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @st.cache
    def load_model(path):
        return torch.load(path,  map_location=torch.device(device))

    #st.markdown('./torch_models/LSTMS_models_'+str(type_ts_)+'/'+country+last_pth)
    lstm_save = load_model('./torch_models/LSTMS_models_'+str(type_ts_)+'/'+country+last_pth)
    #lstm_save = torch.load('./torch_models/LSTMS_models_'+str(type_ts_)+'/'+country+last_pth)
    model = LSTM(seq_length=4,input_size = 1,hidden_size = 4,num_layers = 1,num_classes = 1)
    model.load_state_dict(lstm_save)
    arima_path = './torch_models/ARIMA_models_'+type_ts_+'/'+country+str(type_ts_)+'.pkl'
    fig = plot_ts(model,arima_path, window_to_predict, window, country , type_ts_)
    st.plotly_chart(fig)
    #url = "https://raw.githubusercontent.com/ArtemioPadilla/ML-Datasets/main/Casos_Diarios_Estado_Nacional_Defunciones_20210121.csv"
    #df = pd.read_csv(url)
    #st.dataframe(df.iloc[:,3:])
    #st.line_chart(df.iloc[:,3:])
elif analysis == "About us":
    st.header('Universidad Nacional Autónoma de México')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Alfonso Barajas Cervantes")
        st.markdown("Data Science Major at IIMAS, Applied Mathematics and Systems Research Institute, UNAM.")
        st.image("./imgs/Alfonso_circle.png")

    with col2:
            st.subheader("Artemio Santiago Padilla Robles")
            st.markdown("Physics bachelor graduate and Data Science bachelors student at UNAM")
            st.image("./imgs/Art_circle.png")

    with col3:
        st.subheader("Raul Isaid Mosqueda García")
        st.markdown("Data Science and Actuarial Science undergraduate at IIMAS-UNAM and Science Faculty - UNAM, Mexico.")
        st.image("./imgs/raul_circle.png")

    col11, col21, col31 = st.columns(3)

    with col11:
        st.subheader("Carlos Cerritos Lira")
        st.markdown("Data science undergraduate student at IIMAS, UNAM, Mexico.")
        st.image("./imgs/carlos_circle.png")

    with col21:
        st.subheader("Ingrid Pamela Ruiz")
        st.markdown("Actuarial Science graduate, Data Science student at IIMAS UNAM")
        st.image("./imgs/pamela_ruiz_02-modified.png")

    with col31:
        st.subheader("Guillermo Cota Martínez")
        st.markdown( "Undergraduate student of Data Science at IIMAS, UNAM, Mexico.")
        st.image("./imgs/mem_circle.png")


elif analysis == "Cases and deaths chart":
    with st.expander("See explanation"):
        st.markdown("EXPLANATION FOR LSTMs")
        
        
        st.code("""for i in range(5):
            algorithm""")
    type = st.selectbox("Pick type", ["Cases", "Deaths"])
    st.write(type, 'for all the world countries')
    if type == "Cases":
        fig = px.line(cases_who, x="Date_reported", y="New_cases", color="Country")
    else:
        fig = px.line(cases_who, x="Date_reported", y="New_deaths", color="Country")
    st.plotly_chart(fig)

elif analysis == "Convolutional Neural Networks":
    with st.expander("About this model"):
        st.markdown("The convolutional neural networks on this project used the following architecture:")
        st.image("https://raw.githubusercontent.com/ArtemioPadilla/MachineLearningAndCOVID/main/src/NNClassifiers/cnn.svg")    
        st.markdown("Between each FC layer an nonlinear sigmoid activation function was employed:")
        st.markdown("The variables used to try to detect infected are :")
        st.code("""
sexo, edad, fiebre, tos, odinogia, disnea, irritabilidad, 
diarrea, dolor torácico, calofrios, cefalea, mialgias, artral,
ataque al estado general, rinorrea, polipnea, vomito, dolor abdominal, 
conjuntivitis, cianosis, inicio súbito de sintomas""")

        st.markdown("The variables used to try to detect risk of hospitalization are:")
        st.code("""
sexo,  edad, fiebre, tos, odinogia, disnea,
irritabilidad, diarrea, dolor torácico, calofrios, cefalea, mialgias,
artral, ataque al estado general, rinorrea, polipnea, vomito, 
dolor abdominal, conjuntivitis, cianosis, inicio subito de síntomas, 
diabetes, epoc, asma, inmusupresión, hipertensión, 
vih_sida, otras comorbilidades, enfermedad cardiovascular, obesidad,
insuficiencia renal crónica, tabaquismo""")

        st.markdown("The variables used to try to detect risk of dead are :")
        st.code("""
sexo, edad, diabetes, epoc, asma, inmusupresión, hipertensión, 
vih_sida, otras comorbilidades, enfermedad cardiovascular, obesidad,
insuficiencia renal crónica, tabaquismo""")


        st.markdown(f"The model to try to detect infected was trained on 300,000 patients, and was tested on another 100,000 patients with an average accuracy of {67.60}% the following classification report:")
        st.code("""

______________precision    recall  f1-score   support
    NEGATIVE       0.90      0.66      0.76     78398
       COVID       0.37      0.73      0.49     21602

    accuracy                           0.68    100000
   macro avg       0.63      0.69      0.63    100000
weighted avg       0.78      0.68      0.70    100000

        """)
        st.markdown(f"The model to try to detect hospitalization risk was trained on 258,667 patients, and was tested on another 86,222 patients with an average accuracy of {89.91}% the following classification report:")
        st.code("""
__________________precision    recall  f1-score   support
NOT HOSPITALIZED       0.99      0.90      0.94     81856
    HOSPITALIZED       0.31      0.84      0.46      4367

        accuracy                           0.90     86223
       macro avg       0.65      0.87      0.70     86223
    weighted avg       0.96      0.90      0.92     86223
        """)
        st.markdown(f"The model to try to detect risk of dying was trained on 258,667 patients, and was tested on another 86,222 patients with an average accuracy if {80.56}% and the following classification report:")
        st.code("""
______________precision    recall  f1-score   support
        LIFE       0.99      0.81      0.89     84295
        DEAD       0.09      0.81      0.16      1928

    accuracy                           0.81     86223
   macro avg       0.54      0.81      0.52     86223
weighted avg       0.97      0.81      0.87     86223
""")

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
    
    inisubis = bool_mask[col2.radio("Inicio súbito de síntomas", ["no","sí"])]

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
            if probs[1] > 0.75:
                st.error(f"El clasificador indica que eres POSITIVO a COVID-19 con una seguridad de {probs[1]*100:.2f}%")
            else:
                st.warning(f"El clasificador indica que eres POSITIVO a COVID-19 con una seguridad de {probs[1]*100:.2f}%")
        elif res == 0:
            st.success(f"El clasificador indica que eres NEGATIVO a COVID-19 con una seguridad de {probs[0]*100:.2f}%")



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
            if probs[0] > 0.75:
                st.error(f"Estimamos que si te da COVID-19 van a tener que HOSPITALIZARTE con una seguridad de {probs[1]*100:.2f}%")
            else:
                st.warning(f"Estimamos que si te da COVID-19 van a tener que HOSPITALIZARTE con una seguridad de {probs[1]*100:.2f}%")
        elif res == 0:
            st.success(f"Estimamos que si te da COVID-19 NO van a tener que HOSPITALIZARTE con una seguridad de {probs[0]*100:.2f}%")


        st.markdown("""## Risk of Death""")

        X_death= np.array([[sexo, edad, diabetes, epoc, asma, inmusupr, hiperten,
                            vih_sida, otracon, enfcardi, obesidad, insrencr, tabaquis]])

        res = predict_model(classifier_death, X_death, device)
        probs = prob_model(classifier_death, X_death, device) 

        st.markdown("If you where to have covid given your symptoms and your commorbidities the chances of you dying are:")
        if res == 1:
            if probs[1] > 0.75:
                st.error(f"La red neuronal piensa que vas a MORIR con una seguridad de {probs[1]*100:.2f}%")
            else:
                st.warning(f"La red neuronal piensa que vas a MORIR con una seguridad de {probs[1]*100:.2f}%")
        elif res == 0:
            st.success(f"La red neuronal piensa que vas a SOBREVIVIR con una seguridad de {probs[0]*100:.2f}%")
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
