// Databricks notebook source
// MAGIC %md
// MAGIC # Ricardo López Rodríguez A01066515

// COMMAND ----------

// MAGIC %md ### Standardizing a dataset
// MAGIC 
// MAGIC For this activity you will apply a common standardization technique used as a preprocessing step for most machine learning tasks. Standardizing the variable (column) \\(X\\) consists in computing its standard deviation \\(X_S\\) and mean \\(X_M \\). Then, the standard version \\(X'\\) will be computed by applying the following transformation to each record \\(i\\) of \\(X\\) as per:
// MAGIC 
// MAGIC \\(X^{\prime}_i = \frac {X_i-X_M} {X_S}\\)
// MAGIC 
// MAGIC The goal is to compute the statistics (mean and standard deviation) and standardize each of the following columns in the `/FileStore/tables/day.csv` dataset: 
// MAGIC 
// MAGIC - atemp
// MAGIC - temp
// MAGIC - hum
// MAGIC - windspeed
// MAGIC 
// MAGIC However, as we are using the train-test split strategy for the machine learning model, we need to compute the statistics **ONLY** for the train set, and apply them in **BOTH** train and test sets. For example, for the column `temp` you will compute its mean temperature and standard deviation only for the records in the train set. Next, you will apply those values (a single number for the mean and a single number for the standard deviation) to standardize both train and test records.
// MAGIC 
// MAGIC The train set is defined as all the records in which the column `dteday` \\(\lt\\) "2012-01-01". Any other record will belong to the test set.

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

//Creating the standarized version of both dataSets ... 
//Computing the mean and std deviation of each variable in the train set 
var trainMean = trainSet.agg(avg($"atemp"), avg($"temp"),avg($"hum"),avg($"windspeed"))
var trainStd = trainSet.agg(stddev($"atemp"), stddev($"temp"), stddev($"hum"), stddev($"windspeed"))
//Joinig  both sets with the computed statistics ...
trainSet = trainSet.withColumn("dummy", lit(-1000.974)) //Dummy column for the join operation 
trainSet = trainSet.join(trainMean, trainSet("dummy") =!= trainMean("avg(temp)"), "outer") //outer join 
trainSet = trainSet.join(trainStd, trainSet("dummy") =!= trainStd("stddev_samp(temp)"), "outer") //outer join 

//Standarized variables in the train set...
var standarizedTrainSet = trainSet.select($"dteday",
                                  (($"atemp" - $"avg(atemp)")/$"stddev_samp(atemp)").alias("stand_atemp"),
                                  (($"temp" - $"avg(temp)")/$"stddev_samp(temp)").alias("stand_temp"),
                                  (($"hum" - $"avg(hum)")/$"stddev_samp(hum)").alias("stand_hum"),
                                  (($"windspeed" - $"avg(windspeed)")/$"stddev_samp(windspeed)").alias("stand_windspeed"),
                                  $"casual",$"registered")

//Computing the starized values for the test records ...
testSet = testSet.withColumn("dummy", lit(-1000.123))
testSet = testSet.join(trainMean, testSet("dummy") =!= trainMean("avg(temp)"), "outer")
testSet = testSet.join(trainStd, testSet("dummy") =!= trainStd("stddev_samp(temp)"), "outer")

var standarizedTestSet = testSet.select($"dteday".alias("dteday_test"),
                                       (($"atemp" - $"avg(atemp)")/$"stddev_samp(atemp)").alias("stand_atemp_test"),
                                       (($"temp" - $"avg(temp)")/$"stddev_samp(temp)").alias("stand_temp_test"),
                                       (($"hum" - $"avg(hum)")/$"stddev_samp(hum)").alias("stand_hum_test"),
                                       (($"windspeed" - $"avg(windspeed)")/$"stddev_samp(windspeed)").alias("stand_windspeed_test"),
                                       $"casual",$"registered")


//Saving the standarized datasets ... 
standarizedTrainSet.write.format("csv").mode("overwrite").option("header",true).save("/FileStore/tables/standarizedTrainSet")
standarizedTestSet.write.format("csv").mode("overwrite").option("header",true).save("/FileStore/tables/standarizedTestSet")
display(standarizedTrainSet)

// COMMAND ----------

// MAGIC %md Now train a linear regression model using a Pandas UDF using the **standardized versions** of `atemp`, `temp`, `hum` and `windspeed` as variables to predict a new column called `ratio` which is the division between `casual` and `registered`, i.e. \\(\frac{casual}{registered}\\). Display the coefficients obtained.

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.functions import col, pandas_udf, PandasUDFType
// MAGIC import pyspark.sql.types as T
// MAGIC from sklearn.linear_model import LinearRegression
// MAGIC import pandas as pd
// MAGIC import numpy as np
// MAGIC 
// MAGIC #Function that computes a linear equation from a pandas dataFrame, and returs coeficients and
// MAGIC # intercept in a dataFrame
// MAGIC @pandas_udf("atemp_coeff double, temp_coeff double, hum_coeff double, windspeed_coeff double, intercept double", PandasUDFType.GROUPED_MAP)
// MAGIC def linearRegressionFunction(pdf):
// MAGIC   X = pdf[["stand_atemp","stand_temp","stand_hum", "stand_windspeed"]] #predictors
// MAGIC   Y = pdf[["ratio"]] #predicted variable
// MAGIC   model = LinearRegression(fit_intercept=True, normalize=False).fit(X,Y) #Fit the model
// MAGIC 
// MAGIC   #Extraction of coefficients ...
// MAGIC   results = dict(atemp_coeff =model.coef_[0][0], temp_coeff = model.coef_[0][1],
// MAGIC                  hum_coeff = model.coef_[0][2], windspeed_coeff = model.coef_[0][3],
// MAGIC                  intercept = model.intercept_[0])  
// MAGIC   
// MAGIC   return pd.DataFrame(results, index = [0])
// MAGIC ############################################################################################################################################
// MAGIC train_set = spark.read.format("csv").option("header",True)\
// MAGIC                 .load("/FileStore/tables/standarizedTrainSet") #reading train set
// MAGIC test_set = spark.read.format("csv").option("header",True)\
// MAGIC                 .load("/FileStore/tables/standarizedTestSet") #reading test set
// MAGIC #Variable to be predicted and predictors ... 
// MAGIC train_set_new = train_set.select((col("casual")/col("registered")).alias("ratio"), 
// MAGIC                                  "stand_atemp","stand_temp","stand_hum","stand_windspeed" )
// MAGIC test_set = test_set.select((col("casual")/col("registered")).alias("ratio"), 
// MAGIC                                  "stand_atemp_test","stand_temp_test","stand_hum_test","stand_windspeed_test" )
// MAGIC # Get the coefficients for the train set
// MAGIC coeffs = train_set_new.groupBy().apply(linearRegressionFunction)
// MAGIC #Coefficients
// MAGIC display(coeffs)

// COMMAND ----------

// MAGIC %md Use the coefficients to predict the `ratio` on the test set. Compute the Mean Absolute Error (MAE) and the Mean Absolute Percentage Error (MAPE) of your predictions versus the real values. Display both metrics.

// COMMAND ----------

// MAGIC %md
// MAGIC \\( MAE = \frac{1}{n} \Sigma_{i = 1}^n |y_i  -  x_i|  \\)
// MAGIC 
// MAGIC \\( MAPE = \frac{100}{n} \Sigma_{t = 1}^n |\frac{ A_t  -  F_t }{A_t}| \\)

// COMMAND ----------

// MAGIC %python
// MAGIC #Prediction of temperature 
// MAGIC #Whe use an outter join in order to create dummy columns with the coefficients. 
// MAGIC #Then, we compute de predicted temperature value using a linear equation
// MAGIC 
// MAGIC prediction_df =  test_set.join(coeffs,coeffs.windspeed_coeff != test_set.stand_temp_test ,"outer")\
// MAGIC       .withColumn("predicted_ratio", col("intercept")+ col("stand_hum_test") * col("hum_coeff")+ col("stand_windspeed_test") * col("windspeed_coeff") 
// MAGIC                   + col("stand_atemp_test")* col("atemp_coeff") + col("stand_temp_test")* col("temp_coeff"))\
// MAGIC       .drop("hum_coeff","windspeed_coeff","intercept","atemp_coeff","temp_coeff")
// MAGIC 
// MAGIC display(
// MAGIC   prediction_df.withColumn("difference_abs", F.abs(col("predicted_ratio") - col("ratio")) )\
// MAGIC                 .withColumn("diff_and_ratio", F.abs( 100*((col("ratio") - col("predicted_ratio"))/col("ratio"))))\
// MAGIC                 .select(F.avg(col("difference_abs")).alias("MAE"),F.avg(col("diff_and_ratio")).alias("MAPE") )
// MAGIC )

// COMMAND ----------

// MAGIC %md ### Stratified Sampling
// MAGIC 
// MAGIC Take a look at [this](https://en.wikipedia.org/wiki/Stratified_sampling) definition of stratified sampling from Wikipedia. The objective is to create a **Scala function** that returns a stratified sample of the flights dataset:
// MAGIC 
// MAGIC     /databricks-datasets/learning-spark-v2/flights/departuredelays.csv
// MAGIC 
// MAGIC The dataset consists in the delay time (minutes) of several flights. The first task is to categorize the column `delay` into 4 groups following the next conditions:
// MAGIC 
// MAGIC |     condition     |      group     |
// MAGIC |:-------------:|:--------------:|
// MAGIC |    delay < -15    |    too_early   |
// MAGIC | -15 <= delay <= 0 |      early     |
// MAGIC |  0 < delay <= 30  | tolerably_late |
// MAGIC |     delay > 30    |    too_late    |

// COMMAND ----------

// Write your answer here. The new column's name should be group.
var data_delays = spark.read.format("csv").option("header",true).load("/databricks-datasets/learning-spark-v2/flights/departuredelays.csv")
                       .withColumn("group",  when($"delay" < -15, "too_early").otherwise(
                       when($"delay"<= 0 && $"delay" >= -15, "early").otherwise(
                       when($"delay"<=30 && $"delay" > 0, "tolerably_late").otherwise(
                       when($"delay" > 30, "too_late")))))


display(data_delays)

// COMMAND ----------

// MAGIC %md The stratify function should be implemented using base Scala Spark, i.e. you cannot use any implementation that already performs a stratification. The function should receive 2 arguments: 1) the dataframe to which the function will be applied, and 2) the size of the resulting, stratified dataset. We'll start by explaining an example with dummy data.
// MAGIC 
// MAGIC Let's define our dataset \\(H\\) as a list of names (who are they?) and whether they have hair or not. 
// MAGIC 
// MAGIC |        Name        | Hair |
// MAGIC |:------------------:|:----:|
// MAGIC |     Fei-Fei Li     |  No  |
// MAGIC |   Ilya Sutskever   |  No  |
// MAGIC |     Durk Kingma    |  No  |
// MAGIC |    Rachel Thomas   |  No  |
// MAGIC | Vladimir Iglovikov |  No  |
// MAGIC |    Satya Nadella   |  No  |
// MAGIC |   Ian Goodfellow   |  Yes |
// MAGIC |   Sylvain Gugger   |  Yes |
// MAGIC |    Thomas Kipff    |  Yes |
// MAGIC |   Andrej Karpathy  |  No  |
// MAGIC 
// MAGIC Note that there are 10 records in \\(H\\). In this case, we have a hair distribution of:
// MAGIC 
// MAGIC | Hair | Percentage |
// MAGIC |:----:|:----------:|
// MAGIC |  Yes |     30%    |
// MAGIC |  No  |     70%    |
// MAGIC 
// MAGIC We will subsample the dataset and end up with \\(H_s\\) that has a lenght \\(s\\), defined as the `size` parameter in the Scala function. For this example, let us define `size=6`.
// MAGIC 
// MAGIC A naïve way to subsample is to just apply a `limit(6)` function to get the first 6 records out of the orginal 10. However, we will lose the `Yes` hair cateogry because there won't be any records of people **with** hair. Here is where the stratify method comes into play. By making the subsampling in a stratified manner, we will keep the same relation of Yes:No labels in the result. The following table shows exactly how many records per category we need to keep to still have the original label distribution across categories:
// MAGIC 
// MAGIC | Hair    | Samples Requiered | Rounded labels |
// MAGIC |:-------:|:-----------------:|:--------------:|
// MAGIC |  Yes    |        1.8        |       2        |
// MAGIC |  No     |        4.2        |       5        |
// MAGIC |  Total  |        6          |       7        |
// MAGIC 
// MAGIC Note that to get **exactly** 6 records at the end, we need incomplete records. In this case we should apply a `ceil` function to round up. Now that we have the exact number of records per category, we proceed to sample out of each category in a **random** way.
// MAGIC 
// MAGIC |        Name        | Hair |
// MAGIC |:------------------:|:----:|
// MAGIC |   Ian Goodfellow   |  Yes |
// MAGIC |    Rachel Thomas   |  No  |
// MAGIC | Vladimir Iglovikov |  No  |
// MAGIC |    Thomas Kipff    |  Yes |
// MAGIC |     Fei-Fei Li     |  No  |
// MAGIC |     Durk Kingma    |  No  |
// MAGIC |    Satya Nadella   |  No  |
// MAGIC 
// MAGIC It is very important to sample RANDOMLY inside each category to avoid having a skewed result.
// MAGIC 
// MAGIC A few final remarks:
// MAGIC - The Scala function should not be a UDF nor a public implementation. It should be your own implementation of stratified sampling.
// MAGIC - The Scala function should receive two things: the original dataframe and the size parameter. 
// MAGIC - The Scala function should sample randomly from each category.
// MAGIC - The `departure` dataset has 4 categories, therefore the proportion of each of the 4 labels should be kept in the resulting, subsampled dataframe.

// COMMAND ----------

// Write your answer here.
import org.apache.spark.sql.DataFrame

def stratified_sampling(df:DataFrame ,size:Int):DataFrame  = {
  //Computing the number or records per category
  var df_samples = df.groupBy("group").count()
                 .withColumn("fraction", $"count"/df.count())
                 .withColumn("percentage", $"fraction"*100)
                 .withColumn("samples_requiered", $"fraction"* size)
                 .withColumn("final_samples_requiered", ceil($"samples_requiered")) //rounded samples
  //sampling each category in a random way 
  var df_2 = df.withColumn("group_2",$"group").drop("group")
  var df_3 = df_2.join(df_samples, df_2("group_2") === df_samples("group"), "inner")
  var final_df = df_3.withColumn("row_number", row_number() over Window.partitionBy("group").orderBy(rand())) //shuffle dataSet by group
  
  
  final_df.drop("group_2","fraction","count","samples_requiered","percentage")
          .filter($"row_number" <= $"final_samples_requiered")
  
  
}

// COMMAND ----------

// MAGIC %md
// MAGIC In the following cell, we show the distribution per label in the original dataset, in order to  have a reference for comparison against the results of the next "problem"

// COMMAND ----------

//Distribution per label of the ORIGINAL dataSet
display(
  data_delays.groupBy("group").count()
                   .withColumn("fraction", $"count"/data_delays.count())
                   .withColumn("percentage", $"fraction"*100))

// COMMAND ----------

// MAGIC %md To test your function, please display 1) the distribution per label (in %), and 2) the total count of records of the resulting dataframe using the next `size` values:
// MAGIC - 100
// MAGIC - 25,000
// MAGIC - 1,123,456

// COMMAND ----------

// SIZE = 100
var stratified_sampling_1 = stratified_sampling(data_delays,100) //stratified sample of the flights dataset
display(
  
  stratified_sampling_1.groupBy("group").count()
                   .withColumn("fraction", $"count"/stratified_sampling_1.count())
                   .withColumn("percentage %", $"fraction"*100)
                   .withColumn("dataFrame Size", lit(stratified_sampling_1.count()))
                   .drop("count","fraction")
  )


// COMMAND ----------

//SIZE = 25,000
var stratified_sampling_1 = stratified_sampling(data_delays,25000) //stratified sample of the flights dataset
display(
  
  stratified_sampling_1.groupBy("group").count()
                   .withColumn("fraction", $"count"/stratified_sampling_1.count())
                   .withColumn("percentage %", $"fraction"*100)
                   .withColumn("dataFrame Size", lit(stratified_sampling_1.count()))
                   .drop("count","fraction")
  )

// COMMAND ----------

// SIZE = 1,123,456
var stratified_sampling_1 = stratified_sampling(data_delays,1123456) //stratified sample of the flights dataset
display(
  
  stratified_sampling_1.groupBy("group").count()
                   .withColumn("fraction", $"count"/stratified_sampling_1.count())
                   .withColumn("percentage %", $"fraction"*100)
                   .withColumn("dataFrame Size", lit(stratified_sampling_1.count()))
                   .drop("count","fraction")
  )

// COMMAND ----------

