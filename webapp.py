import streamlit as st
import pandas as pd
from math import pow

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


# ---------------- SIDEBAR -------------------
with st.sidebar:
    st.title('Configuración datos')
    hints_enabled = st.checkbox(label='Mostrar descripciones',
                                value=True)
    st.header('Carga de datos')

    raw_file = st.file_uploader(label='Conjunto de datos (*.xlsx, *.csv)',
                                type=['.csv', '.xlsx'])
    if hints_enabled:
        st.markdown("""
                    *Sube aquí el archivo con tu conjunto de datos, puede ser un
                    fichero de excel (\*.xlsx) o un archivo CSV (\*.csv)*
                    """)

    if raw_file:
        df = pd.read_csv(raw_file) if raw_file.name.endswith('.csv') \
                                   else pd.read_excel(raw_file)\

        st.header('Selección de columnas')
        if hints_enabled:
            st.markdown("""
                        Selecciona las columnas que quieres incluir para el
                        estudio que se va a realizar posteriormente, si se
                        deja vacío, se incluirán todas las columnas
                        """)

        features_selected = st.multiselect(label='Columnas para el estudio estadístico',
                                           options=df.columns,
                                           max_selections=2)

        enough_features = len(features_selected) == 2

        if not enough_features:
            st.error("""
                    Se requieren 2 columnas para realizar el estudio
                    estadístico correctamente.
                    """, icon='❕')
        else:
            df = df.loc[:, features_selected] if features_selected else df

            st.header('Selección de variables')
            if hints_enabled:
                st.markdown("""
                            Selecciona de las variables de tu conjunto de datos la
                            que será la variable independiente (x) y la variable
                            dependiente (y) en el modelo.
                            """)

            x = st.selectbox(label='Variable X',
                             options=df.columns)

            y = st.selectbox(label='Variable Y',
                             options=df.columns)

if raw_file and enough_features:
    st.header('Análisis de los datos')
    st.subheader('Tablas de variable bidimensional')
    st.markdown("""
                En este apartado se podrán encontrar las tablas de frecuencia y
                doble entrada de variable bidimensional acorde a las variables
                elegidas dentro de los datos importados. *Para importar un
                conjunto de datos, dirígete al panel lateral de la página*.
                """)
    # Introduce all the logic related to those tables

    st.subheader('Métricas estadísticas')
    st.markdown("""
                En este apartado se puede encontrar el valor de las métricas
                estadísticas correspondientes al estudio de la variable
                bidimensional.
                """)

    mean_x = df[x].mean()
    mean_y = df[y].mean()

    std_x = df[x].std(ddof=0)
    std_y = df[y].std(ddof=0)

    var_x = pow(df[x].std(ddof=0), 2)
    var_y = pow(df[y].std(ddof=0), 2)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 style="text-align: center; padding: 0;">Métricas variable X</h3>', unsafe_allow_html=True)

        st.latex(r'\Large\bar{x}=' + str(round(mean_x, 3)))
        st.latex(r'\Large\sigma_x=' + str(round(std_x, 3)))
        st.latex(r'\Large\sigma^2_x=' + str(round(var_x, 3)))

    with col2:
        st.markdown('<h3 style="text-align: center; padding: 0;">Métricas variable Y</h3>', unsafe_allow_html=True)

        st.latex(r'\Large\bar{x}=' + str(round(mean_y, 3)))
        st.latex(r'\Large\sigma_x=' + str(round(std_y, 3)))
        st.latex(r'\Large\sigma^2_x=' + str(round(var_y, 3)))

    st.markdown('<h3 style="text-align: center;">Métricas de X e Y</h3>', unsafe_allow_html=True)

    cov_xy = df.cov(ddof=0)[x][y]
    r2 = cov_xy / (std_x * std_y)

    st.latex(r'\Large\sigma_{xy}=' + str(round(cov_xy, 4)))
    st.latex(r'\Large r=' + str(round(r2, 5)))
