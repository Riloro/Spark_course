# Databricks notebook source
# MAGIC %md 
# MAGIC ### Getting familiar with Databricks
# MAGIC 
# MAGIC As an introductory excersise, today we will perform a simple analysis on a Kaggle dataset. The steps consist on:
# MAGIC 
# MAGIC 1. Download the train and test csvs from Kaggle's [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists).
# MAGIC 2. Import both files inside the folder `/FileStore/tables/kaggle_HR_dataset/` with their original filenames.
# MAGIC 3. Display the raw data using Pandas.
# MAGIC 4. Using seaborn, __PLOT__ the answer to the following questions:
# MAGIC   - How does the `relevent_experience` relates to the `education_level`?
# MAGIC   - Who is looking for a job change? Relationship between the `gender` column and `target` column.
# MAGIC   - From those who have a `relevent experience`, how many are looking for a job change?
# MAGIC 5. Create a Dashboard to shows your results and export it in HTML format. This file will be graded.
# MAGIC 
# MAGIC For help on displaying a Matplotlib figure in databricks, please refer to the [documentation](https://docs.databricks.com/notebooks/visualizations/matplotlib.html).

# COMMAND ----------

# MAGIC %md
# MAGIC #Reclutamiento de nuevos empleados
# MAGIC ##### Ricardo López Rodriguez A01066515         19/08/2021
# MAGIC 
# MAGIC Las siguientes figuras fueron generadas para responder preguntas específicas sobre reclutamiento de
# MAGIC empleados que podría tener una empresa. El conjunto de datos fue obtenio de Kagle, y las figuras fueron generadas
# MAGIC con la biblioteca Plotly. 

# COMMAND ----------

#%fs file system
#%md Markdown
#%sh shell
#%run Corrrer notebook

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns

spark_df = spark.read.format("csv").option("header", True).load("/FileStore/tables/kaggle_HR_dataset/aug_train.csv")

pandas_df = spark_df.toPandas()

# COMMAND ----------

sns.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ##### A continuación se muestra la tabla con los datos crudos que serán filtrados y tratados para reponder a las preguntas planteadas

# COMMAND ----------

display(spark_df)

# COMMAND ----------

spark_df.count()

# COMMAND ----------

pandas_df

# COMMAND ----------

#Convirtiendo las variables cualitativas a numéricas ...
dummie_experience = pd.get_dummies(pandas_df["relevent_experience"], dummy_na=False)
dummie_education = pd.get_dummies(pandas_df["education_level"], dummy_na=False)
#Concatenando las variables dummies
pd_df = pd.concat([pandas_df,dummie_experience, dummie_education], axis = 1)

# COMMAND ----------

pd_df #Nuevo dataFrame

# COMMAND ----------

#Filtrando los que tienen experiencia relevante 
relevant_exp = pd_df[pd_df["Has relevent experience"] == 1]
#Filtrando los que no tienen experiencia relevante
no_relevant_exp = pd_df[pd_df["No relevent experience"] == 1]

# COMMAND ----------

relevant_exp #Con experiencia relevante

# COMMAND ----------

#Hacer un conteo de personas por nivel educativo ....
relevant_exp_edu = relevant_exp[["Graduate","High School","Masters","Phd","Primary School"]]


# COMMAND ----------

edu_level_1 = relevant_exp_edu.sum().to_frame() #Sumatoria de cada columna 
edu_level_1 = edu_level_1.sort_values(by = 0, ascending = False)

# COMMAND ----------

import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

# COMMAND ----------

#Desplegando el total de personas en cada nivel educativo
pandas_df["education_level"].value_counts()

# COMMAND ----------

fig_1 = px.bar(edu_level_1, title= "Personas con experiencia relevante")
fig_1.update_layout(width= 650, height = 550)
fig_1.show()

# COMMAND ----------

no_relevant_exp

# COMMAND ----------

#Hacer un conteo de personas por nivel educativo ....
no_relevant_exp_edu = no_relevant_exp[["Graduate","High School","Masters","Phd","Primary School"]]

# COMMAND ----------

edu_level_2 = no_relevant_exp_edu.sum().to_frame() #Sumatoria de cada columna 
edu_level_2 = edu_level_2.sort_values(by = 0, ascending = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # ¿Cómo la experiencia relevante se relaciona con el nivel de educación? 

# COMMAND ----------

import plotly.graph_objects as go

# COMMAND ----------

# MAGIC %md
# MAGIC A continuació se muestran dos tablas, la primera es el número de personas con experiencia relevante y la segunda el número de personas sin experiencia por nivel educativo

# COMMAND ----------

edu_level_1

# COMMAND ----------

edu_level_2

# COMMAND ----------

fig_bars = go.Figure(data=[
  go.Bar(name = "Personas con experiencia", x =edu_level_1.index, y= edu_level_1[0], marker = dict(color = "#0288d1") ),
  go.Bar(name = "Personas sin experiencia", x = edu_level_2.index, y = edu_level_2[0], marker = dict(color = "#ff5f52"))
])
fig_bars.update_layout(title_text = "Número de personas con experiencia y sin experiencia relevante categorizadas por nivel educativo",
                      width = 1300, height= 700)

fig_bars.update_yaxes(title= "Número de personas")
fig_bars.show()

# COMMAND ----------

# MAGIC %md
# MAGIC La gráfica anterior muestra como el mayor número de personas con experiencia tienen un nivel educativo de *graduate* , *masters* es el segundo nivel educativo con una gran catidad de personas con experiencia. El gráfico también nos muestra como hay mayor número de jóvenes de *High School* con experiencia laboral que adultos con un grado de *Phd*, estos resultados podrían ser debidos a que hay un gran número de "Phd's" que se dedican a la investigación en instituciones académicas, gubernamentales o privadas, mientras que los jóvenes tienen tiempo de estudiar y realizar un trabajo. Además, como era de esperarse, el nivel educativo *Primary School* contiene el menor número de personas con experiencia relevante.

# COMMAND ----------

#Graficando el conteo de personas sin experiencia

#Realizar gráfica de barramasters
fig_2 = px.bar(edu_level_2, title= "Personas sin experiencia relevante")
fig_2.update_layout(width= 650, height = 550)
fig_2.show()

# COMMAND ----------

px.data.tips()

# COMMAND ----------

#Buscando la relacion entre target y gender 
#target: 0 – Not looking for job change, 1 – Looking for a job change
pd_df[["gender", "target"]]
pd_df_gender_tar = pd.get_dummies(pd_df["gender"], dummy_na=False) 
df_gender_target = pd.concat([pd_df_gender_tar, pd_df["target"]],axis = 1)

# COMMAND ----------

#Filtrando las personas que buscan un cambio de trabajo
df_gender_target
#Concatenando con el df original 
pd_df = pd_df.drop("target", axis=1)
df_pd = pd.concat([df_gender_target,pd_df], axis = 1)
df_pd

# COMMAND ----------

corte_gender = df_gender_target[df_gender_target["target"] == "1.0"]
df_gender_cambio = corte_gender.drop("target",axis = 1)
df_gender_cambio = df_gender_cambio.sum().to_frame() #Sumatoria de cada columna 
df_gender_cambio = df_gender_cambio.sort_values(by = 0, ascending = False)

# COMMAND ----------

df_gender_cambio.rename(columns={0: "sexo"}, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # ¿Quién está buscando un cambio de trabajo?

# COMMAND ----------

df_gender_cambio

# COMMAND ----------

fig_3 = px.bar(df_gender_cambio, title= "Personas que buscan un cambio de trabajo")
fig_3.update_layout(width= 650, height = 550)
fig_3.update_traces(marker=dict(color= "#0288d1"))
fig_3.update_xaxes(title= "Sexo")
fig_3.update_yaxes(title= "Número de personas")
fig_3.show()

# COMMAND ----------

# MAGIC %md
# MAGIC La gráfica anterior muestra el número de personas que buscan un cambio de trabajo, categorizadas por sexo. Para realizar el gráfico se filtraron todas aquellas personas que tuvieran un valor de *1* en la variable *target*. Podemos concluir que la mayoria de personas que buscan un cambio de trabajo son del sexo masculino. La clase del sexo masculino cuenta con un total de 3000 personas, miestras que solo 326 *mujeres* buscan un cambio de trabajo, y solo 50 personas de la clase *otros* .

# COMMAND ----------

# MAGIC %md
# MAGIC # De aquellos que tienen experiencia relevante, ¿cuántos están buscando un cambio de trabajo?

# COMMAND ----------

#From those who have a relevent experience, how many are looking for a job change?
df_pd_new = df_pd[df_pd["Has relevent experience"] == 1]
df_pd_new = df_pd_new["target"]
target = df_pd_new.value_counts().to_frame()

# COMMAND ----------

#Graficando las personas con experiencia y que buscan un cambio de trabajo
variables = ["No buscan un cambio de trabajo","Buscan un cambio de trabajo"]
fig_4 = px.bar(x= variables, y= target, title= "Decisión sobre la búsqueda de trabajo de personas con experiencia")
fig_4.update_layout(width= 600, height = 550)
fig_4.update_traces(marker = dict(color = "#ff5f52"))
fig_4.update_xaxes(title = "")
fig_4.update_yaxes(title = "Número de personas")
fig_4.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Después de hacer algunos filtros a los datos crudos, la gráfica anterior muestra que el total de personas con experiencia relevante y que buscan un cambio de trabajo son 2961, mientras 
# MAGIC 10,831 personas no buscan un cambio de empleo. 
# MAGIC 
# MAGIC Además se muestra la tabla con la que se realizo la gráfica, donde se tomó la variable traget = 1 como las peronas que buscan el cambio de trabajo y target = 0 para las personas que no buscan el cambio.

# COMMAND ----------

target