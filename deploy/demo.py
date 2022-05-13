import streamlit as st
import pandas as pd
from time import sleep
# import lstm
# import classifier

st.title("Some applications of deep learning for the COVID-19 pandemic")


st.markdown("In this application you can either get forecast for the COVID-19 pandemic infected number using **LSTMs** or you can use a **neural network classifier** to get probabilities of desease and hospitalization for an individual with certain characteristics")

st.markdown("more info to be written here")

analysis = st.radio("Please enter which type of application you want to explore:", ["LSTM forecast", "Neural Network Classifier"])

if analysis == "LSTM forecast":
    st.markdown("EXPLANATION FOR LSTMs")
    st.code("""for i in range(5):
        asdasd""")
    st.metric(label = "Test", value=1, delta=-0.1, delta_color="normal")
    # Example plot
    url = "https://raw.githubusercontent.com/ArtemioPadilla/ML-Datasets/main/Casos_Diarios_Estado_Nacional_Defunciones_20210121.csv"
    df = pd.read_csv(url)
    #st.dataframe(df.iloc[:,3:])
    st.line_chart(df.iloc[:,3:])
else:
    st.markdown("EXPLANATION FOR CLASSIFIER")

    st.markdown("Please enter your symptoms")
    col1, col2, col3 = st.columns(3)
    fiebre = col1.radio("Fiebre", ["sí", "No"])
    tos = col2.radio("Tos", ["sí", "No"])
    dolordecuerpo = col3.radio("Dolor de cuerpo", ["sí", "No"])
    s4 = col1.radio("sintoma4", ["sí", "No"])
    s4 = col2.radio("sintoma5", ["sí", "No"])
    s4 = col3.radio("sintoma6", ["sí", "No"])

    st.markdown("Please enter your symptoms")

    col1, col2, col3 = st.columns(3)
    fiebre = col1.radio("com1", ["sí", "No"])
    tos = col2.radio("com2", ["sí", "No"])
    dolordecuerpo = col3.radio("com3", ["sí", "No"])
    s4 = col1.radio("com4", ["sí", "No"])
    s4 = col2.radio("com5", ["sí", "No"])
    s4 = col3.radio("com6", ["sí", "No"])
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