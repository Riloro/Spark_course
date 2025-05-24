// Databricks notebook source


// COMMAND ----------

// MAGIC %md 
// MAGIC ## Pivots, Arrays, High-level functions, Window
// MAGIC 
// MAGIC Using the data in /FileStore/tables/day.csv:
// MAGIC   1. Transform the dataset so that each record has the day and the windspeed for each week.
// MAGIC   2. Create a column array that contains the windspeed of each day of the current week.
// MAGIC   3. Obtain the max and min windspeed of the current week in another column.
// MAGIC   4. Create another column that contains all the windspeed records of the week, but the max and min.
// MAGIC   5. Create a dataframe that has, for each week, the delta of max temperatures compared with the previous week.

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

// COMMAND ----------

var dataSet = spark.read.format("csv").option("header",true).load("/FileStore/tables/day.csv")
             .select("dteday","windspeed")
             .withColumn("day",to_date($"dteday", "yyyy-MM-dd"))
             .drop("dteday")
             .withColumn("week",weekofyear($"day"))
             .filter($"day" >"2011-01-02" && $"day"< "2021-12-31")
             .withColumn("year",year($"day"))
             .withColumn("week", when($"week" < 10, concat(lit("0"), $"week".cast(StringType))).otherwise($"week".cast(StringType)))
              .withColumn("week_2",concat($"year".cast(StringType),$"week"))
              .select("windspeed","day","week")
              .withColumn("windspeed",$"windspeed".cast(DoubleType))
display(
  dataSet.withColumn("day_string",date_format($"day","E"))
          .groupBy("week").pivot("day_string", values = Array("Mon","Tue", "Wed", "Thu","Fri","Sat","Sun")).sum("windspeed")
          .withColumn("total_array",array($"Mon",$"Tue", $"Wed", $"Thu",$"Fri",$"Sat",$"Sun"))
          .withColumn("array_max_min", array(expr("array_max(total_array)"), expr("array_min(total_array)")))
          .withColumn("all_but_max_min", expr("array_except(total_array,array_max_min)"))
       )

// COMMAND ----------

var temp = spark.read.format("csv").option("header",true).load("/FileStore/tables/day.csv")
             .select("dteday","temp")
             .withColumn("day",to_date($"dteday", "yyyy-MM-dd"))
             .drop("dteday")
             .withColumn("week",weekofyear($"day"))
             .filter($"day" >"2011-01-02" && $"day"< "2021-12-31")
             .withColumn("year",year($"day"))
             .withColumn("week", when($"week" < 10, concat(lit("0"), $"week".cast(StringType))).otherwise($"week".cast(StringType)))
              .withColumn("week", concat($"year".cast(StringType),$"week"))
              .groupBy("week").agg(max("temp").alias("max_temp"))
display(
  temp.withColumn("max_temp_lw", lag($"max_temp",1) over Window.partitionBy().orderBy($"week".asc))
  .withColumn("delta",$"max_temp"- $"max_temp_lw")
  .select("week","delta")
)

// COMMAND ----------

Lun  -        Mar   -   Mie    - Jue   - vie   - sab  -  dom  -  week    -  array         - subarray   -  others
0.160446   0.248539                                             201101      [x,x,x,x,x]      [y,y]        [x,x,x,x,x]
                                                                201201


week      max_temp     delta
201101       1         null
201102       2         2-1 = 1

// COMMAND ----------

// MAGIC %md ## Homework
// MAGIC 
// MAGIC 1. What is the difference between sortBy and orderBy in spark?
// MAGIC 2. How can one effectively create a column that contains the consecutive number of record for each row? Can monotonically_incresing_id be used?

// COMMAND ----------

