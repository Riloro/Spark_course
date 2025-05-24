// Databricks notebook source
// MAGIC %md ## Data Cleaning and basic processing using Scala and Spark.
// MAGIC 
// MAGIC Perform a series of actions and transformations to the dataset in `/databricks-datasets/asa/planes/plane-data.csv`

// COMMAND ----------

// MAGIC %md
// MAGIC ## Ricardo López Rodríguez A01066515 
// MAGIC 
// MAGIC #### 13/09/2021

// COMMAND ----------

var df1 = spark.read.format("csv").option("header",true).load("/databricks-datasets/asa/planes/plane-data.csv")
display(df1)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Data Cleaning

// COMMAND ----------

// MAGIC %md ##### 1) Remove _null_ and _None_ values.

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


// COMMAND ----------

// Write your answer here 
//Comenzando a eliminar los valores nulos del dataFrame 
df1  = df1.na.drop() //Drop rows containing nulls in any columns 

for(c <- df1.columns){
  df1 = df1.filter(col(c) =!= "None")
}
display(df1)

// COMMAND ----------

// MAGIC %md ##### 2) Create column `status_boolean` by transforming the column status to boolean.

// COMMAND ----------

// Write your answer here
var df2 = df1.withColumn("status_boolean", when($"status" === "Valid", true).otherwise(false) )
display(df2)

// COMMAND ----------

// MAGIC %md ##### 3) Impute column year using column `issue_date` and subtract 1 year for EMBRAER and 2 years for AIRBUS INDUSTRIE.

// COMMAND ----------

// Write your answer here
var df3 =
  df2.withColumn("issue_date", to_date($"issue_date","MM/dd/yyy"))
      .withColumn("issue_date_year", year($"issue_date"))
      .withColumn("year",when($"year" === "0000" && $"manufacturer" === "EMBRAER", $"issue_date_year"-1).otherwise(
      when($"year" === "0000" && $"manufacturer" === "AIRBUS INDUSTRIE", $"issue_date_year"-2).otherwise($"year")))
      .drop("issue_date_year")

// COMMAND ----------

// MAGIC %md ##### 4) Create column `year_diff` with the number of days between the column `year` and `issue_date` (absolute value).

// COMMAND ----------

// Write your answer here
var df4 = df3.withColumn("year_date", to_date($"year","yyyy"))
          .withColumn("year_diff", datediff($"issue_date",$"year"))
display(df4)

// COMMAND ----------

// MAGIC  %md ##### 5) Remove the "Turbo-" string from the column `engine_type`.

// COMMAND ----------

// Write your answer here -------> regexp_replace
 var df5 = df4.withColumn("engine_type",regexp_replace(df4("engine_type"),"Turbo-",""))
display(df5)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Processing

// COMMAND ----------

// MAGIC %md ##### 1) Top 3 years with more BOEING issues of Fixed Wing Multi-Engine.

// COMMAND ----------

// Write your answer here

//We'll use the issue_date column to group issues by year
var df6 = df5.withColumn("issue_year", year($"issue_date"))
//Filtering ...
df6 = df6.filter($"aircraft_type" === "Fixed Wing Multi-Engine" && $"manufacturer" === "BOEING")
//Grouping by issue_year and counting the number of row per group ....
// Ordering from higher to lower values of $count ...
df6 = df6.groupBy("issue_year").count()
          .orderBy(desc("count")).limit(3)

display(df6)

// COMMAND ----------

// MAGIC %md ##### 2) Plot the amount of planes issued per manufacturer.

// COMMAND ----------

// Write your answer here

//Grouping by manufacturer
var df7 = df5.groupBy("manufacturer").count().orderBy(desc("count"))
display(df7)


// COMMAND ----------

// MAGIC %md ##### 3) Which model had the most issues? Which had the least?

// COMMAND ----------

//Grouping by plane model ...
var df_temp=  df5.groupBy("model").count()
                  .orderBy(desc("count"))

//Filtering ... Max Value = 383 and Min value = 1
var least_iss = df_temp.filter($"count" === 1)
var most_iss = df_temp.filter($"count" === 383)
//Union
var df_final = most_iss.union(least_iss)
//Display of the models with the most and least isssues ....
display(df_final)

// COMMAND ----------

// MAGIC %md ##### 4) Is there a relationship between the number of issues and the year of fabrication? How does it relate to the amount of planes?

// COMMAND ----------

// Write your answer here
// Grouping by year and counting the number of issues ...
var df_by_year = df5.groupBy("year").count().orderBy("year")
df_by_year = df_by_year.filter($"year" =!= "0000") //droping 0000 "years"
display(df_by_year)

// COMMAND ----------



// COMMAND ----------

// MAGIC %md
// MAGIC La grafica anterior muestra como los problemas comenzaron a aumentar en los aviones que fueron manufacturados en los inicios del siglo XXI o en los años cercanos a este siglo(1998 y 1999).Con un pico de 389 problemas registrados en el año 2001, después de este año 
// MAGIC los problemas reportados comenzaron a disminuir hasta llegar a niveles similares que se tenian en los años 80s y 90s.