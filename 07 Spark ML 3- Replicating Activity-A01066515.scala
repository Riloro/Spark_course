// Databricks notebook source
// MAGIC %md ### Alternative to UDFs
// MAGIC 
// MAGIC Replicate the first part of the [Standardization and Stratification](https://drive.google.com/file/d/1LmThdjctTBRb1Hlx0lGRv6kPXYppuzXs/view?usp=sharing) assignment using SparkML features. 
// MAGIC 
// MAGIC You will predict the ratio column using a **linear regression** and the **standardized** version of `temp`, `atemp`, `hum`, and `windspeed` on the same definition of test-train split. Please show the resulting coefficients and intercept of the model.
// MAGIC 
// MAGIC After having a [standardizer](https://spark.apache.org/docs/latest/ml-features.html#standardscaler) and [linear regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression) objects from SparkML, create a [Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html) that concatenates them into a single transformer. Save the pipeline on disk and show that the predictions are the same before and after being saved.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Ricardo López Rodríguez A01066515

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

// COMMAND ----------

//Reading and casting data ...
var data = spark.read.format("csv").option("header",true).load("/FileStore/tables/day.csv")
             .select($"dteday", $"atemp".cast(DoubleType), $"temp".cast(DoubleType), $"hum".cast(DoubleType),
                     $"windspeed".cast(DoubleType),$"casual".cast(DoubleType),$"registered".cast(DoubleType))
             .withColumn("dteday", to_date($"dteday", "yyyy-MM-dd"))
//Spliting data in train and test set  ...  
var trainSet = data.filter($"dteday" < "2012-01-01") 
var testSet = data.filter($"dteday" >= "2012-01-01")
//add ratio colum to both datasets
trainSet = trainSet.withColumn("ratio", col("casual")/col("registered"))
testSet = testSet.withColumn("ratio", col("casual")/col("registered"))
display(trainSet)

// COMMAND ----------

//Using Vector Assembler
import org.apache.spark.ml.feature.VectorAssembler

val vectorAssembler = new VectorAssembler()
                      .setInputCols(Array("atemp","temp","hum","windspeed"))
                      .setOutputCol("features")
val vecTrainDF = vectorAssembler.transform(trainSet)

display(vecTrainDF)

// COMMAND ----------

//Standarizing the dataSet
import org.apache.spark.ml.feature.StandardScaler
//StandardScaler 
//uses the unbiased sample standard deviation
val scaler = new StandardScaler()
              .setInputCol("features")
              .setOutputCol("scaled_features")
              .setWithStd(true)
              .setWithMean(true)
//Summaty statistics
val scalerModel = scaler.fit(vecTrainDF)
// Mean zero and unit standard deviation
val scaledData = scalerModel.transform(vecTrainDF)
display(scaledData)

// COMMAND ----------

//Implementation of a linear regression
import org.apache.spark.ml.regression.LinearRegression

val model = new LinearRegression()
            .setFeaturesCol("scaled_features")
            .setLabelCol("ratio")
val regression = model.fit(scaledData)

// COMMAND ----------

println("Resulting coefficients and intercept:")
val m = regression.coefficients
val b = regression.intercept

// COMMAND ----------

// MAGIC %md
// MAGIC Por los coeficientes calculados en la celda anterror, podemos escribir nuestro modelo de regresión como:
// MAGIC \\( ratio = 0.1175197048167617 \times atemp -0.049182759581750925 \times temp -0.023317427429179267 \times hum -0.010732606410627432 \times windspeed +  0.2478961341091216    \\)

// COMMAND ----------

//Pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel}

val pipeline = new Pipeline().setStages(Array(vectorAssembler,scaler,model))
val pipeModel = pipeline.fit(trainSet)

// COMMAND ----------

// MAGIC %md
// MAGIC ##Predictions before saving

// COMMAND ----------

//Testing 
val predictions_pipeline = pipeModel.transform(testSet)
display(predictions_pipeline.drop("casual","registered","features"))

// COMMAND ----------

//saving to disk
pipeModel.write.overwrite().save("/tmp/spark-linear-regression-model")

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Predictions after saving 

// COMMAND ----------

//load back fitted pipeline
val load_model = PipelineModel.load("/tmp/spark-linear-regression-model")

//Testing 
val predictions_pipeline_loaded = load_model.transform(testSet)
display(predictions_pipeline_loaded.drop("casual","registered","features"))

// COMMAND ----------

// MAGIC %md
// MAGIC Predictions are the same before and after saving the fitted pipeline

// COMMAND ----------

