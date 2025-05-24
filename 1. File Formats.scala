// Databricks notebook source
// DBTITLE 1,Dataset information
// MAGIC %fs head /databricks-datasets/flights/README.md

// COMMAND ----------

// DBTITLE 1,Location
// MAGIC %fs ls /databricks-datasets/flights/departuredelays.csv

// COMMAND ----------

display(
  spark.read.format("csv").option("header",true).load("/databricks-datasets/flights/departuredelays.csv")
       )

// COMMAND ----------

// DBTITLE 1,Read
var original = spark.read.format("csv") // Define a format
                    .option("header", true) // Headers?
                    .load("/databricks-datasets/flights/departuredelays.csv") // Path

// COMMAND ----------

display(original)

// COMMAND ----------

// MAGIC %fs mkdirs /users/practica1

// COMMAND ----------

// DBTITLE 1,Write dataset
original//.coalesce(1)
        .write
        .format("csv") // Format
        .mode("overwrite") // Save type
        .option("header", true)
        .save("/users/practica1/dist_csv") // Path

// original -> dist
// original -> original *no es necesario
// original -> avro
// original -> parquet

// avro -> procesar (8) -> display
// parquet -> procesar (8)  -> display
// original -> procesar (8) -> display
// dist -> procesar (8) -> display

// COMMAND ----------

original//.coalesce(1)
        .write
        .format("avro") // Format
        .mode("overwrite") // Save type
        .option("header", true)
        .save("/users/practica1/dist_avro") // Path

// original -> dist
// original -> original *no es necesario
// original -> avro
// original -> parquet

// avro -> procesar (8) -> display
// parquet -> procesar (8)  -> display
// original -> procesar (8) -> display
// dist -> procesar (8) -> display

// COMMAND ----------

original//.coalesce(1)
        .write
        .format("parquet") // Format
        .mode("overwrite") // Save type
        .option("header", true)
        .save("/users/practica1/dist_parquet") // Path

// original -> dist
// original -> original *no es necesario
// original -> avro
// original -> parquet

// avro -> procesar (8) -> display
// parquet -> procesar (8)  -> display
// original -> procesar (8) -> display
// dist -> procesar (8) -> display

// COMMAND ----------

// MAGIC %fs ls /users/practica1/dist_csv

// COMMAND ----------

// MAGIC %fs ls /users/practica1/dist_avro

// COMMAND ----------

// MAGIC %fs ls /users/practica1/dist_parquet

// COMMAND ----------

spark.read.format("csv").option("header", true).load("/users/practica1/dist_csv").count()


// COMMAND ----------

// DBTITLE 1,Delete files
// MAGIC %fs rm -r /users/practica1/dist_avro

// COMMAND ----------

// DBTITLE 1,Sum
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

display(
       spark.read.format("csv").option("header", true).load("/databricks-datasets/flights/departuredelays.csv")
            .agg(sum("delay"), avg("distance"))
)

// COMMAND ----------

val load_csv =  spark.read.format("csv").option("header", true).load("/users/practica1/dist_csv")

// COMMAND ----------

   spark.read.format("csv").option("header", true).load("/databricks-datasets/flights/departuredelays.csv").agg(sum("delay")).show()

// COMMAND ----------

display(
       load_csv.filter($"delay">0)
)

// COMMAND ----------

display(
       spark.read.format("csv").option("header", true).load("/databricks-datasets/flights/departuredelays.csv")
            .withColumn("nueva", $"delay" + $"distance")
)

// COMMAND ----------

// MAGIC %md 
// MAGIC ### The time it takes to display the average of delay and distance columns in the same command.

// COMMAND ----------

val load_parquet = spark.read.format("parquet").option("header", true).load("/users/practica1/dist_parquet")

// COMMAND ----------

display(load_parquet.agg(avg("delay"),avg("distance")))

// COMMAND ----------

val load_avro = spark.read.format("avro").option("header",true).load("/users/practica1/dist_avro")

// COMMAND ----------

display(load_csv.agg(avg("delay"), avg("distance")))

// COMMAND ----------

display(load_parquet.agg(avg("delay"),avg("distance")))

// COMMAND ----------

display(original.agg(avg("delay"),avg("distance")))

// COMMAND ----------

display(load_avro.agg(avg("delay"),avg("distance")))

// COMMAND ----------

// MAGIC %md
// MAGIC ### The time it takes to filter using one column: keep records where the column delay<5

// COMMAND ----------

display(load_csv.filter($"delay" < 5))

// COMMAND ----------

display(load_avro.filter($"delay" < 5))

// COMMAND ----------

display(load_parquet.filter($"delay" < 5))

// COMMAND ----------

display(original.filter($"delay" < 5))

// COMMAND ----------

// MAGIC %md
// MAGIC ### The time it takes to filter in several columns: keep records where the column delay>0 and distance >500

// COMMAND ----------

display(load_csv.filter($"delay" > 0 && $"distance" > 500))

// COMMAND ----------

display(load_avro.filter($"delay" > 0 && $"distance" > 500))

// COMMAND ----------

display(load_parquet.filter($"delay" > 0 && $"distance" > 500))

// COMMAND ----------

display(original.filter($"delay" > 0 && $"distance" > 500))

// COMMAND ----------

