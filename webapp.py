import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def gen_freq_table(df):
    unique_pairs = df.value_counts().index.to_numpy()
    frequencies = df.value_counts()\
                    .to_numpy()\
                    .reshape(-1, 1)

    grades_array = []
    for pair in unique_pairs:
        x, y = pair
        grades_array.append(x)
        grades_array.append(y)

    unique_pairs = np.array([grades_array]).reshape(-1, 2)
    pair_freq_data = np.concatenate((unique_pairs, frequencies), axis=1)

    return (pd.DataFrame(pair_freq_data, columns=['x_i', 'y_i', 'n_i']),
            unique_pairs)


def gen_double_entry_tbl(freq_table, unique_pairs):
    unique_xs = list(set(unique_pairs[:, 0]))
    unique_ys = list(set(unique_pairs[:, 1]))

    print(unique_xs)

    array = np.ndarray((len(unique_ys), len(unique_xs)),
                       dtype='int8')
    array.fill(0)

    double_entry_table = pd.DataFrame(array,
                                      columns=unique_xs,
                                      index=unique_ys)
    for x, y in unique_pairs:
        double_entry_table.loc[y, x] = freq_table.loc[(freq_table['x_i'] == x) &
                                                      (freq_table['y_i'] == y)]\
                                                      .iloc[0, -1]
    double_entry_table['Total'] = 0

    for i in double_entry_table.index:
        double_entry_table.loc[i, 'Total'] = np.sum(double_entry_table.loc[i])

    new_index = double_entry_table.index.join(pd.Index(['Total']), how='outer')
    double_entry_table = double_entry_table.reindex(index=new_index,
                                                    fill_value=0)

    for col in double_entry_table:
        double_entry_table.loc['Total', col] = double_entry_table[col].sum()

    return double_entry_table


def covariance_state(cov_val):
    if cov_val >= -0.3 and cov_val <= 0.3:
        return r'\sigma_{xy}\approx 0'
    elif cov_val < -0.3:
        return r'\sigma_{xy}\lt 0'
    else:
        return r'\sigma_{xy}\gt 0'


def corr_coef_state(r2):
    if r2 <= -0.8 and r2 >= -0.9:
        return r'r\approx -1'
    elif r2 == -1:
        return 'r=-1'
    elif r2 >= 0.8 and r2 <= 0.9:
        return r'r\approx 1'
    elif r2 == 1:
        return 'r=1'
    if r2 >= -0.3 and r2 <= 0.3:
        return r'r\approx 0'
    else:
        return r'r\ll 1' if r2 > 0 else r'r\gg -1'


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
        st.markdown(r"""
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

# ----------- FREQ TABLES -----------
if raw_file and enough_features:
    st.header('Análisis de los datos')
    st.subheader('Tablas de variable bidimensional')
    st.markdown("""
                En este apartado se podrán encontrar las tablas de frecuencia y
                doble entrada de variable bidimensional acorde a las variables
                elegidas dentro de los datos importados. *Para importar un
                conjunto de datos, dirígete al panel lateral de la página*.
                """)

    rel_freq, double_entry = st.columns(2)

    freq_df, unique_pairs = gen_freq_table(df)

    with rel_freq:
        st.markdown('<p style="font-size: 20px;">\
                    Tabla de frecuencias relativas</p>',
                    unsafe_allow_html=True)

        st.dataframe(freq_df)

    with double_entry:
        st.markdown('<p style="font-size: 20px;">\
                    Tabla de doble entrada</p>', unsafe_allow_html=True)
        st.dataframe(gen_double_entry_tbl(freq_df, unique_pairs))

# ----------- STAT MEASURES -----------

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

    var_x = df[x].std(ddof=0)**2
    var_y = df[y].std(ddof=0)**2

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 style="text-align: center; padding: 0;">\
                     Métricas variable X</h3>', unsafe_allow_html=True)

        st.latex(r'\Large\bar{x}=' + str(round(mean_x, 3)))
        st.latex(r'\Large\sigma_x=' + str(round(std_x, 3)))
        st.latex(r'\Large\sigma^2_x=' + str(round(var_x, 3)))

    with col2:
        st.markdown('<h3 style="text-align: center; padding: 0;">\
                     Métricas variable Y</h3>', unsafe_allow_html=True)

        st.latex(r'\Large\bar{y}=' + str(round(mean_y, 3)))
        st.latex(r'\Large\sigma_y=' + str(round(std_y, 3)))
        st.latex(r'\Large\sigma^2_y=' + str(round(var_y, 3)))

    st.markdown('<h3 style="text-align: center;">\
                 Métricas de X e Y</h3>', unsafe_allow_html=True)

    cov_xy = df.cov(ddof=0)[x][y]
    r2 = cov_xy / (std_x * std_y)

    st.latex(r'\Large\sigma_{xy}=' + str(round(cov_xy, 4)))
    st.latex(r'\Large r=' + str(round(r2, 5)))

# ----------- LINREG -----------

    st.subheader('Diagrama de dispersión')
    st.markdown("""
                En este apartado podrás observar la relación entre las métricas
                de la covarianza y el coeficiente de correlación de Pearson y
                la disposición de los puntos en la nube de puntos.
                """)

    st.pyplot(plt.scatter(data=df, x=y, y=x).figure)
    st.latex(covariance_state(cov_xy) + ';' + corr_coef_state(r2))

# ----------- LINREG -----------

    st.subheader('Regresión Lineal')
    st.markdown("""
                En este apartado se muestran los diagramas de dispersión con
                las respectivas rectas de regresión de una variable sobre la
                otra.
                """)

    xy_regr_col, yx_regr_col = st.columns(2)

    with xy_regr_col:
        st.markdown('<p style="text-align: center; font-size: 20px;"><strong>\
                    Recta de regresión de y sobre x</strong></p>',
                    unsafe_allow_html=True)

        xy_model = LinearRegression()
        xy_model.fit(df[x].to_numpy().reshape(-1, 1), df[y])

        fig, ax = plt.subplots()
        ax.scatter(data=df, x=x, y=y, c='blue')
        ax.axline((0, xy_model.intercept_), slope=xy_model.coef_[0], c='blue')

        plt.xlim(df[x].min())
        plt.ylim(df[y].min())

        st.pyplot(fig)
        st.latex(r'y=' + str(round(xy_model.coef_[0], 3)) + 'x'
                 + '+' + str(round(xy_model.intercept_, 2)))

    with yx_regr_col:
        st.markdown('<p style="text-align: center; font-size: 20px;"><strong>\
                    Recta de regresión de y sobre x</strong></p>',
                    unsafe_allow_html=True)

        yx_model = LinearRegression()
        yx_model.fit(df[y].to_numpy().reshape(-1, 1), df[x])

        fig, ax = plt.subplots()
        ax.scatter(data=df, x=y, y=x, c='red')
        ax.axline((0, yx_model.intercept_), slope=yx_model.coef_[0], c='red')

        plt.xlim(df[y].min())
        plt.ylim(df[x].min())

        st.pyplot(fig)

        st.latex(r'x=' + str(round(yx_model.coef_[0], 3)) + 'y'
                 + '+' + str(round(yx_model.intercept_, 2)))

# ----------- LINREG TESTING -----------

    st.subheader('Aplicación de las rectas de regresión')
    st.markdown("""
                Finalmente, en este apartado puedes introducir valores
                para cada una de las variables, de modo que se realice
                una estimación utilizando la ecuación de las rectas de
                regresión calculadas:
                """)

    x_pred_col, y_pred_col = st.columns(2)

    with x_pred_col:
        x_val = st.number_input(label='Valor X',
                                min_value=0.00,
                                max_value=10.00,
                                step=1.00)

        y_pred = xy_model.predict([[x_val]])
        st.latex(r'x=' + str(x_val))

        st.latex(r'y=' + str(round(xy_model.coef_[0], 3)) + r'\cdot'
                 + str(x_val) + '+' + str(round(xy_model.intercept_, 2)) + '='
                 + str(round(y_pred[0], 2)))

    with y_pred_col:
        y_val = st.number_input(label='Valor Y',
                                min_value=0.00,
                                max_value=10.00,
                                step=1.00)

        x_pred = yx_model.predict([[y_val]])
        st.latex(r'y=' + str(y_val))

        st.latex(r'x=' + str(round(yx_model.coef_[0], 3)) + r'\cdot'
                 + str(y_val) + '+' + str(round(yx_model.intercept_, 2)) + '='
                 + str(round(x_pred[0], 2)))
