import json
import folium
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from PIL                      import Image
from plotly                   import express as px
from warnings                 import filterwarnings
from folium.plugins           import MarkerCluster
from streamlit_folium         import folium_static
from matplotlib.pyplot        import figimage
from distutils.fancy_getopt   import OptionDummy
# filterwarnings('ignore')
# st.set_page_config(page_title='App - Pronóstico',
#                     layout="wide", 
#                     page_icon='🚀',  
#                     initial_sidebar_state="expanded")

# st.title("Pronosticando precios de casas")
# st.sidebar.markdown("Características")

# # @st.cache
# def get_data(allow_output_mutation=True):
#     url = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/kc_house_data.csv'
#     # url = 'kc_house_data.csv'
#     return pd.read_csv(url)

# data = get_data()
# datta = data.copy()
# datta['price/sqft'] = datta['price']/datta['sqft_living']
# datta['year_old'] = 2020-datta['yr_built']
# datta = datta.drop(columns=['price'])
# ### banios info
# banhos = st.sidebar.select_slider(
#           'Número de Baños',
#           options=list(sorted(set(datta['bathrooms']))), value=0.75)

# ### habitaciones info
# habitaciones = st.sidebar.number_input('Número de habitaciones', min_value=1, max_value=11, value=2, step=1)

# ### area info
# area = st.sidebar.number_input('Área del inmueble', value=1020)

# ### area de troncos
# area_lote = st.sidebar.number_input('Área de lote', value=1076)

# ### pisos info
# pisos = st.sidebar.select_slider(
#           'Número de Pisos',
#           options=list(sorted(set(datta['floors']))), value=2)

# ### info vista al mar: si/no
# waterfront = st.sidebar.selectbox(
#      '¿Vista al agua?',
#      ('Sí', 'No'))

# if waterfront == 'Sí': 
#     waterfront = 1
# else:  
#     waterfront = 0

# ### info calidad de la vista
# vista = st.sidebar.selectbox(
#      'Puntaje de la vista',
#      (0,1,2,3,4))

# ### info estado de la casa
# condicion = st.sidebar.selectbox(
#      'Condición del inmueble',
#      (1, 2, 3, 4, 5))

# ### info grado de constuccion
# puntaje =  st.sidebar.selectbox(
#      'Puntaje de construcción',
#      (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))

# ### area 1 info
# area_arriba = st.sidebar.number_input('Área sin sótano', value=1020)

# ### area 2 info
# area_abajo = st.sidebar.select_slider(
#           'Área del sótano',
#           options=list(sorted(set(datta['sqft_basement']))), value=0)

# ### code info
# zipcode = st.sidebar.select_slider(
#           'Codigo postal de la vivienda',
#           options=list(sorted(set(datta['zipcode']))), value=98144)

# ### info edad de la casa
# edad = st.sidebar.number_input('Edad', min_value=1, max_value=120, value=12, step=1)

# ### info renovacion si/no
# renovacion = st.sidebar.selectbox(
#      '¿Renovación?',
#      ('Sí', 'No'))

# if renovacion == 'Sí': 
#     renovacion = 1
# else:  
#     renovacion = 0

# ### asiganacion de valores del vector
# X = pd.DataFrame()

# X.loc[0,'bedrooms'] = habitaciones
# X.loc[0,'bathrooms'] = banhos
# X.loc[0,'sqft_living'] = area
# X.loc[0,'sqft_lot'] = area_lote
# X.loc[0,'floors'] = pisos
# X.loc[0,'waterfront'] = waterfront
# X.loc[0,'view'] = vista
# X.loc[0,'condition'] = condicion
# X.loc[0,'grade'] = puntaje
# X.loc[0,'sqft_above'] = area_arriba
# X.loc[0,'sqft_basement'] = area_abajo
# X.loc[0,'zipcode'] = zipcode
# X.loc[0,'year_old'] = edad
# X.loc[0,'renovated_status'] = renovacion

# ### informacion por pantalla
# st.markdown("""
# En esta pestaña, un modelo de Machine Learning ha sido disponibilizado para generar pronósticos de precios  basado en las propuidades del inmueble. El usuario deberá suministrar las características de tal inmueble utilizando el menú de la barra izquierda. A continuación se definen la información requerida. :
     
# - Número de baños: Número de baños de la propiedad a sugerir precio. Valores como 1.5 baños se refiere a la existencia de un baño con ducha y un baño sin dicha. 
# - Número de habitaciones: Número de habitaciones de la propiedad a sugerir precio
# - Área del inmueble: Área en pies cuadrados de la propiedad a sugerir precio
# - Área de lote: Área en pies cuadrados del terreno.
# - Número de pisos: Número de pisos de la propiedad a sugerir precio
# - Vista al agua: La propiedad a sugerir precio tiene vista al agua?
# - Puntaje de la vista: Puntaje de la vista de la propiedad a sugerir precio.
# - Condición del inmueble: Condición general de la propiedad a sugerir precio.
# - Puntaje sobre la construcción: Puntja sobre la construcción de la propiedad a sugerir precio
# - Área sin sótano: Área en pies cuadrados de la vivienda sin contar el sótano.
# - Área del sótano: Área en pies cuadrados del sótano de la vivienda.
# - Código postal de la vivienda: Número de identificación de cada vivienda.
# - Edad de la propiedad: La antiguedad de la propiedad a sugerir precio. 
# - Renovación: La propiedad a sugerir precio ha sido renovada?
#     """)

# variables = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#              'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 
#              'zipcode', 'year_old', 'renovated_status']

# params = {'Habitaciones':['bedrooms', habitaciones],
#           'Baños':['bathrooms', banhos],
#           'Pisos':['floors', pisos],
#           'Edad':['year_old', edad]
#          }

# OptFiltro = st.multiselect(
#      'Variables a incluir en los filtros:',
#      ['Habitaciones', 'Baños', 'Pisos', 'Edad'],
#      ['Baños'])



# col1, col2 = st.columns(2)

# with col1:

#     data_v2 = datta.copy()
#     for filtro in OptFiltro:
#         (llave, variable) = params[filtro]
#         data_v2 = data_v2[data_v2[llave]==variable]

#     data_v2['zipcode'] = data_v2['zipcode'].astype(str)

#     st.header("Ubicación y detalles de casas disponibles acorde a los requerimientos del cliente.")
#     mapa = folium.Map(location=[data_v2['lat'].mean(), data_v2['long'].mean()], zoom_start=9)
#     markercluster = MarkerCluster().add_to(mapa)
#     for nombre, fila in data_v2.iterrows():
#         folium.Marker([fila['lat'],fila['long']],
#                          popup = 'Fecha: {} \n {} habitaciones \n {} baños \n constuida en {} \n área de {} pies cuadrados \n Precio por pie cuadrado: {}'.format(
#                          fila['date'],
#                          fila['bedrooms'],
#                          fila['bathrooms'],
#                          fila['yr_built'], 
#                          fila['sqft_living'], 
#                          fila['price/sqft'])
#           ).add_to(markercluster)
#     folium_static(mapa)

# col1, col2 = st.columns(2)
# with col1:

#     data_v3 = datta.copy()
#     data_v3['zipcode'] = data_v3['zipcode'].astype(str)
#     st.header("Costo de pie cuadrado por código postal")
#     data_aux = data_v3[['price/sqft','zipcode']].groupby('zipcode').mean().reset_index()
#     custom_scale = (data_aux['price/sqft'].quantile((0,0.2,0.4,0.6,0.8,1))).tolist()

#     mapa = folium.Map(location=[data_v3['lat'].mean(), data_v3['long'].mean()], zoom_start=8)
#     url2 = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/KingCount.geojson'
#     folium.Choropleth(geo_data=url2, 
#                         data=data_aux,
#                         key_on='feature.properties.ZIPCODE',
#                         columns=['zipcode', 'price/sqft'],
#                         threshold_scale=custom_scale,
#                         fill_color='YlOrRd',
#                         highlight=True).add_to(mapa)
#     folium_static(mapa)

# # OptFiltro = st.multiselect(
# #      'Variables a incluir en los filtros:',
# #      ['Habitaciones', 'Baños', 'Área construida (pies cuadrados)','Pisos','Vista al agua','Evaluación de la propiedad','Condición'],
# #      ['Habitaciones', 'Baños'])

# ### se carga el model xgboost para la estimacion del valor de la casa
# ### se muestra por panatalla
# if st.sidebar.button('Los parámetros han sido cargados. Calcular precio'):

#     modelo_final = pickle.load(open('model_x.sav', 'rb'))
#     vector = np.array(list(X.loc[0])).reshape(-1, 1).T
#     precio = modelo_final.predict(vector)[0]
#     st.balloons()
#     st.success('El precio ha sido calculado')
#     # st.write('El precio sugerido es:', )
#     # st.metric("Precio Sugerido", np.expm1(precio), str(list(X.loc[0])))
#     st.metric("Precio Estimado:", f"${np.float(round(precio, 2))}")

#     st.header(f"Un total de {data_v2.shape[0]} casas coinciden con las caracteristicas requeridas por el usuario.")
#     st.dataframe(data_v2)  

# else:
#     st.snow()
#     st.error('Por favor, seleccione los parámatros de la propiedad a estimar el precio.')
