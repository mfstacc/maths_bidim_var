import streamlit as st
from pandas import read_csv, read_excel

# ------------ MAIN PAGE -----------------

st.title('Analizador de notas')
st.markdown("""
            <details>
            <summary style="font-size: 22px">
                <strong>Información sobre la web</strong>
            </summary>
            <p>
            Esta página tiene por objetivo constituir un sistema de análisis
            que tome por entrada un conjunto de datos en el que se contemplen
            dos clases diferenciadas (<em>particularmente, está enfocado a las
            calificaciones de los alumnos en dos materias distintas, en dos
            trimestres diferentes o ambas a la vez</em>) y genere las métricas
            propias de la estadística bidimensional además de un modelo de
            estimación teniendo en cuenta tanto una nota como la otra de
            variable independiente.
            </p>
            </details>
            """, unsafe_allow_html=True)

st.header('Análisis de los datos')
st.markdown("""
            ### Tablas de variable bidimensional
            En este apartado se podrán encontrar las tablas de frecuencia y
            doble entrada de variable bidimensional acorde a las variables
            elegidas dentro de los datos importados. Para importar un conjunto
            de datos, dirígete al panel lateral de la página.
            """)

# ---------------- SIDEBAR -------------------
with st.sidebar:
    st.title('Configuración datos')
    hints_enabled = st.checkbox(label='Mostrar descripciones',
                                value=True)
    st.header('Carga de datos')

    raw_file = st.file_uploader(label='Conjunto de datos (*.xlsx, *.csv)',
                                type=['.csv', '.xlsx'])
    st.markdown("""
                *Sube aquí el archivo con tu conjunto de datos, puede ser un
                fichero de excel (\*.xlsx) o un archivo CSV (\*.csv)*
                """)

    if raw_file:
        df = read_csv(raw_file) if raw_file.name.endswith('.csv') else read_excel(raw_file)

        st.header('Selección de variables')
        st.markdown("""
                    Selecciona de las variables de tu conjunto de datos la
                    que será la variable independiente (x) y la variable
                    dependiente (y) en el modelo.
                    """)

        x = st.selectbox(label='Variable X',
                         options=df.columns)

        y = st.selectbox(label='Variable Y',
                         options=df.columns)
