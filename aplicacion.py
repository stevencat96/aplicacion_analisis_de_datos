import json
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt 

from re import template
from PIL import Image
import streamlit as st
import pandas as pd
#import joblib
import numpy as np
#import boto3
import tempfile

from PIL                      import Image
from plotly                   import express as px
from folium.plugins           import MarkerCluster
from streamlit_folium         import folium_static
from matplotlib.pyplot        import figimage
from distutils.fancy_getopt   import OptionDummy



st.set_page_config(page_title='App - Venta de casas',
                    layout="wide", 
                    page_icon=':house',  
                    initial_sidebar_state="expanded")




st.title('Venta de casas en King County')
st.header('Propuesto por [Jose Steven Calderon](https://www.linkedin.com/in/jose-steven-calderon-neira-77a530232/)')


# @st.cache
def get_data():
     url = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/kc_house_data.csv'
     return pd.read_csv(url)

data = get_data()
data_ref = data.copy()

st.sidebar.markdown("# Parámetros a tener en cuenta")
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d').dt.date
data['yr_built']= pd.to_datetime(data['yr_built'], format = '%Y').dt.year
# data['yr_renovated'] = data['yr_renovated'].apply(lambda x: pd.to_datetime(x, format ='%Y') if x >0 else x )
# data['id'] = data['id'].astype(str)

#llenar la columna anterior con new_house para fechas anteriores a 2015-01-01
data['house_age'] = 'NA'
#llenar la columna anterior con new_house para fechas anteriores a 2015-01-01
data.loc[data['yr_built']>2000,'house_age'] = 'new_house' 
#llenar la columna anterior con old_house para fechas anteriores a 2015-01-01
data.loc[data['yr_built']<2000,'house_age'] = 'old_house'

data['zipcode'] = data['zipcode'].astype(str)


data.loc[data['yr_built']>=2000,'house_age'] = 'new_house' 
data.loc[data['yr_built']<2000,'house_age'] = 'old_house'

data.loc[data['bedrooms']<=1, 'dormitory_type'] = 'studio'
data.loc[data['bedrooms']==2, 'dormitory_type'] = 'apartment'
data.loc[data['bedrooms']>2, 'dormitory_type'] = 'house'

data.loc[data['condition']<=2, 'condition_type'] = 'bad'
data.loc[data['condition'].isin([3,4]), 'condition_type'] = 'regular'
data.loc[data['condition']== 5, 'condition_type'] = 'good'

data['price_tier'] = data['price'].apply(lambda x: 'Seccion economica' if x <= 625000 else
                                                   'Seccion media' if (x > 625000) & (x <= 1250000) else
                                                   'seccion media alta' if (x > 1250000) & (x <= 1875000) else
                                                   'seccion alta' )
                                                  

data['price/sqft'] = data['price']/data['sqft_living']

# st.dataframe(data)
st.write('Este es un proyeecto que tiene como finalidad representar de forma facil eficiente y variable in informacion de una data que trata sobre la venta de casas en King County en EEUU, si usted desea descargar o analizar la base de datos de manera detallada y completa podra hacerlos dando click en [este lugar](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) ')



## Filtros
st.subheader('Primera parte (acotamiento de las condiciones)')
# construccion = st.slider('Construcción después de:', int(data['yr_built'].min()),int(data['yr_built'].max()),1991)

#############################################################################################################

st.markdown("""
Las vivienda las he dividido en 4 secciones de la misma extension, basadas en su precio. 
-  En la seccion economica veremos información de las propiedades que cuestan menos de \$625.000 
-  En la seccion media veremos información de las propiedades que cuestan entre \$625.000 y \$1.250.000
-  En la seccion media alta veremos información de las propiedades que cuestan entre \$1.250.000 y \$1.875.000
-  En la seccion alta información de las propiedades que cuestan más de \$1.875.000 
    """)


tier = st.multiselect(
     'Cuartil de precios',
    list(data['price_tier'].unique()),
     list(data['price_tier'].unique()))

#zipcod = st.multiselect(
#     'Códigos postales',
#      list(sorted(set(data['zipcode']))),
#      list(sorted(set(data['zipcode']))))
#data = data[(data['price_tier'].isin(tier))&(data['zipcode'].isin(zipcod))]
#st.subheader('Filtros adicionales (Opcionales)')



############################################################################################################

st.subheader('Filtros acotadores')

data = data[(data['price']<=2500000)&
     (data['bedrooms']<=7)&
     (data['sqft_living']<=3500)&
     (data['sqft_above']<=5000)&
     (data['sqft_basement']<=3000)&
     (data['bathrooms']<=7)&       
     (data['floors']<=3)]


OptFiltro = st.multiselect(
     'Variables a incluir en los filtros:',
     ['Habitaciones', 'Baños', 'Pies cuadrados de sala','Area total de la estructura','Area del terreno','Pisos','Vista al agua','Precio'],
)

if 'Habitaciones' in OptFiltro: 
     if data['bedrooms'].min() < data['bedrooms'].max():
          min_habs, max_habs = st.sidebar.select_slider(
          'Número de Habitaciones',
          options=list(sorted(set(data['bedrooms']))),
          value=(data['bedrooms'].min(),data['bedrooms'].max()))
          data = data[(data['bedrooms']>= min_habs)&(data['bedrooms']<= max_habs)]
     
    
if 'Baños' in OptFiltro: 
     if data['bathrooms'].min() < data['bathrooms'].max():
          min_banhos, max_banhos = st.sidebar.select_slider(
          'Número de baños ',
          options=list(sorted(set(data['bathrooms']))),
          value=(data['bathrooms'].min(), data['bathrooms'].max()))
          data = data[(data['bathrooms']>= min_banhos)&(data['bathrooms']<= max_banhos)]
     else:
          st.markdown("""
               El filtro **Baños** no es aplicable para la selección actual de valores
               """)

if 'Pies cuadrados de sala' in OptFiltro: 
     if data['sqft_living'].min() < data['sqft_living'].max():
          area1 = st.sidebar.slider('Área de la sala', int(data['sqft_living'].min()),int(data['sqft_living'].max()),2000)
          data = data[data['sqft_living']<area1]
     else:  
          st.markdown("""
               El filtro **Pies cuadrados de sala** no es aplicable para la selección actual de valores
               """)

if 'Pisos' in OptFiltro: 
     if data['floors'].min() < data['floors'].max():
          min_pisos, max_pisos = st.sidebar.select_slider(
          'Número de Pisos',
          options=list(sorted(set(data['floors']))),
          value=(data['floors'].min(),data['floors'].max()))
          data = data[(data['floors']>= min_pisos)&(data['floors']<= max_pisos)]
     else:
          st.markdown("""
               El filtro **Pisos** no es aplicable para la selección actual de valores
               """)

if 'Vista al agua' in OptFiltro: 
     if data['view'].min() < data['view'].max():
          min_vista, max_vista = st.sidebar.select_slider(
          'Puntaje de vista al agua',
          options=list(sorted(set(data['view']))),
          value=(data['view'].min(),data['view'].max()))
          data = data[(data['view']>= min_vista)&(data['view']<= max_vista)]
     else:
          st.markdown("""
               El filtro **Vista al agua** no es aplicable para la selección actual de valores
               """)
               
if 'Precio' in OptFiltro:
     if data['price'].min() < data['price'].max():
          min_pric, max_pric = st.sidebar.select_slider(
          'Precio de la propiedad',
          options=list(sorted(set(data['price']))),
          value=(data['price'].min(),data['price'].max()))
          data = data[(data['price']>= min_pric)&(data['price']<= max_pric)]
     else:
          st.markdown("""
               El filtro **Precio** no es aplicable para la selección actual de valores
               """)

if 'Area total de la estructura' in OptFiltro: 
     if data['sqft_above'].min() < data['sqft_above'].max():
          area2 = st.sidebar.slider('Área construida total de la estructura', int(data['sqft_above'].min()),int(data['sqft_above'].max()),3500)
          data = data[data['sqft_above']<area2]
     else:  
          st.markdown("""
               El filtro **Área construida (pies cuadrados)** no es aplicable para la selección actual de valores
               """)

if 'Area del terreno' in OptFiltro: 
     if data['sqft_basement'].min() < data['sqft_basement'].max():
          area3 = st.sidebar.slider('area del terreno', int(data['sqft_basement'].min()),int(data['sqft_basement'].max()),5000)
          data = data[data['sqft_basement']<area3]
     else:  
          st.markdown("""
               El filtro **Área construida (pies cuadrados)** no es aplicable para la selección actual de valores
               """)

          

# # Mapas 

# import plotly.express as px
# import pandas as pd


# # info geojson
# url2 = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/KingCount.geojson'
# col1, col2 = st.columns(2)

# with col1: 
#       st.header("Ubicación y detalles de casas disponibles")
#       mapa = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=9)
#       markercluster = MarkerCluster().add_to(mapa)
#       for nombre, fila in data.iterrows():
#            folium.Marker([fila['lat'],fila['long']],
#                           popup = 'Precio: ${} Fecha: {} \n {} habitaciones \n {} baños \n constuida en {} \n área de {} pies cuadrados \n Precio por pie cuadrado: {}'.format(
#                           fila['price'],
#                           fila['date'],
#                           fila['bedrooms'],
#                           fila['bathrooms'],
#                           fila['sqft_above'], 
#                           fila['sqft_living'], 
#                           fila['price/sqft'])
#            ).add_to(markercluster)
#       folium_static(mapa)

# import plotly.express as px

# with col2:
#      url3 = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/kc_house_data.csv'

     
#      #data= pd.read_csv(url)

#      houses = data.loc[data['price']<=2500000, ['id','lat','long','price']]
#      fig = px.scatter_mapbox(data_frame=houses,
#                               lat = 'lat',
#                               lon = 'long',
#                               size = 'price',
#                               color = 'price') 
#      fig.update_layout(mapbox_style="open-street-map")
#      fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})


#      st.plotly_chart(fig , use_container_width=True)

# # Estadística Descriptiva 
att_num = data.select_dtypes(include = ['int64','float64'])
st.dataframe(att_num)
media = pd.DataFrame(att_num.apply(np.mean))
mediana = pd.DataFrame(att_num.apply(np.median))
std = pd.DataFrame(att_num.apply(np.std))
maximo = pd.DataFrame(att_num.apply(np.max))
minimo = pd.DataFrame(att_num.apply(np.min))

df_EDA = pd.concat([minimo,media,mediana,maximo,std], axis = 1)
df_EDA.columns = ['Mínimo','Media','Mediana','Máximo','std']
st.header('Datos descriptivos')
df_EDA = df_EDA.drop(index =['id', 'lat', 'long','yr_built','yr_renovated','condition','grade','waterfront','sqft_lot15','sqft_living15','price/sqft'], axis = 0 )

df_EDA.index =['Precio','No. Cuartos', 'No. Baños', 'Área construida sala', 
                    'Área del terreno (pies cuadrados)', 'No. pisos', 
                    'Puntaje de la vista', 
                    'Área sobre tierra',  
                    'Área del terreno 15 casas más próximas']
col1, col2 = st.columns(2)
col1.metric("No. Casas", data.shape[0],str(100*round(data.shape[0]/data_ref.shape[0],4))+'% de las casas disponibles',delta_color="off")
col2.metric("No. Casas Nuevas (Construida después de 2000)",data[data['house_age'] == 'new_house'].shape[0],str(100*round(data[data['house_age'] == 'new_house'].shape[0]/data_ref.shape[0],4))+'% de las casas disponibles',delta_color="off")
st.dataframe(df_EDA)  

# st.header('Algunas tendencias')


# col1, col2 = st.columns(2)
# with col1: 
#      st.write('Evolución del precio por tipo de propiedad y año de construcción')
#      data['dormitory_type']=data['bedrooms'].apply(lambda x: 'Estudio' if x <=1 else 'Apartamento' if x==2 else 'Casa' )
#      df = data[['yr_built', 'price','dormitory_type']].groupby(['yr_built','dormitory_type']).mean().reset_index()
#      with sns.axes_style("darkgrid"):
#           plt.style.use('dark_background')
#           fig = plt.figure(figsize=(7,7)) # try different values
#           fig = sns.lineplot(x ='yr_built', y= 'price', data = df, hue="dormitory_type", style="dormitory_type")
#           fig.set_xlabel("Año de Construcción", fontsize = 17)
#           fig.set_ylabel("Precio (Millones de Dólares)", fontsize = 17)
#           fig.legend(title='Tipo de propiedad', loc='upper right', labels=['Apartamento', 'Casa','Estudio'])
#           fig = fig.figure
#           st.pyplot(fig)