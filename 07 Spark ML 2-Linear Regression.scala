// Databricks notebook source
// MAGIC %md ### Spark ML
// MAGIC 
// MAGIC Spark ML is a part of the core utilities that Spark offers. Please use the following guide if you have any concerns: https://spark.apache.org/docs/latest/ml-guide.html}
// MAGIC 
// MAGIC In general, one can break down the process of using Spark ML for machine learning into six stages:
// MAGIC   1. DataFrame: Create train and test datasets.
// MAGIC   2. Transform data (train and test):
// MAGIC       - Apply feature _Transformers_ like Normalize, Scale, OneHotEncoder, StringIndexer, MinMax, Impute.
// MAGIC       - Use _Vector Assembler_ (train and test): A transformer that combines all features into a single column, used as training instance.
// MAGIC   4. Model definition: Select a model and its hyper-parameters.
// MAGIC   5. Train model using train data.
// MAGIC   6. Predict on test data.
// MAGIC 
// MAGIC Other interesting methods to explore are:
// MAGIC - Cross-validation
// MAGIC - Grid Search
// MAGIC 
// MAGIC There are transformers that are maybe easier to implement in code, however the main advantage of having them already there in Spark ML is that you can chain them inside a Pipeline().

// COMMAND ----------

// MAGIC %md
// MAGIC ### Regression: Predicting Rental Price
// MAGIC 
// MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.

// COMMAND ----------

val filePath = "/tmp/sf-airbnb/sf-airbnb-clean.parquet" 
val airbnbDF = spark.read.parquet(filePath)
display(airbnbDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Train/Test Split
// MAGIC 
// MAGIC When we are building ML models, we don't want to look at our test data (why is that?). 
// MAGIC 
// MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set. We will use the `randomSplit` method [Python](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset).
// MAGIC 
// MAGIC **Question**: Why is it necessary to set a seed?

// COMMAND ----------

val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

println(f"There are ${trainDF.cache().count()} rows in the training set, and ${testDF.cache().count()} in the test set")

// COMMAND ----------

// MAGIC %md
// MAGIC We are going to build a very simple linear regression model predicting `price` just given the number of `bedrooms`.

// COMMAND ----------

display(trainDF.select("price", "bedrooms").summary())

// COMMAND ----------

// MAGIC %md
// MAGIC There do appear some outliers in our dataset for the price ($10,000 a night??). Just keep this in mind when we are building our models :).

// COMMAND ----------

// MAGIC %md
// MAGIC ## Vector Assembler
// MAGIC 
// MAGIC Linear Regression expects a column of Vector type as input.
// MAGIC 
// MAGIC We can easily get the values from the `bedrooms` column into a single vector using `VectorAssembler` [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler). VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val vecAssembler = new VectorAssembler()
  .setInputCols(Array("bedrooms"))
  .setOutputCol("features")

val vecTrainDF = vecAssembler.transform(trainDF)

vecTrainDF.select("bedrooms", "features", "price").show(10)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Linear Regression
// MAGIC 
// MAGIC Now that we have prepared our data, we can use the `LinearRegression` estimator to build our first model [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.LinearRegression). Estimators accept a DataFrame as input and return a model, and have a `.fit()` method. 

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
  .setFeaturesCol("features")
  .setLabelCol("price")

val lrModel = lr.fit(vecTrainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Inspect the model

// COMMAND ----------

val m = lrModel.coefficients(0)
val b = lrModel.intercept

println(f"The formula for the linear regression line is price = $m%1.2f*bedrooms + $b%1.2f")
println("*-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Pipeline

// COMMAND ----------

import org.apache.spark.ml.Pipeline

val pipeline = new Pipeline().setStages(Array(vecAssembler, lr))

val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Apply to Test Set

// COMMAND ----------

val predDF = pipelineModel.transform(testDF)

predDF.select("bedrooms", "features", "price", "prediction").show(10)

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Save the pipeline, read it again and verify that the predicted values are the same.

// COMMAND ----------

