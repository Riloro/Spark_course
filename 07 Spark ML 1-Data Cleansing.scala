// Databricks notebook source
// MAGIC %md
// MAGIC # Data Cleansing with Airbnb
// MAGIC 
// MAGIC We're going to start by doing some exploratory data analysis & cleansing. We will be using the SF Airbnb rental dataset from [Inside Airbnb](http://insideairbnb.com/get-the-data.html).

// COMMAND ----------

// MAGIC %md
// MAGIC Let's load the SF Airbnb dataset (comment out each of the options if you want to see what they do).

// COMMAND ----------

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb.csv"

val rawDF = spark.read
  .option("header", "true")
 .option("multiLine", "true")
  .option("inferSchema", "true")
  .option("escape", "\"")
  .csv(filePath)

display(rawDF)

// COMMAND ----------

rawDF.columns

// COMMAND ----------

// MAGIC %md
// MAGIC For the sake of simplicity, only keep certain columns from this dataset. We will talk about feature selection later.

// COMMAND ----------

val baseDF = rawDF.select(
  "host_is_superhost",
  "cancellation_policy",
  "instant_bookable",
  "host_total_listings_count",
  "neighbourhood_cleansed",
  "latitude",
  "longitude",
  "property_type",
  "room_type",
  "accommodates",
  "bathrooms",
  "bedrooms",
  "beds",
  "bed_type",
  "minimum_nights",
  "number_of_reviews",
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value",
  "price")

baseDF.cache().count
display(baseDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Fixing Data Types
// MAGIC 
// MAGIC Take a look at the schema above. You'll notice that the `price` field got picked up as string. For our task, we need it to be a numeric (double type) field. 
// MAGIC 
// MAGIC Let's fix that.

// COMMAND ----------

import org.apache.spark.sql.functions.translate

val fixedPriceDF = baseDF.withColumn("price", translate($"price", "$,", "").cast("double"))

display(fixedPriceDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Summary statistics
// MAGIC 
// MAGIC Two options:
// MAGIC * describe
// MAGIC * summary (describe + IQR)

// COMMAND ----------

display(fixedPriceDF.describe())

// COMMAND ----------

display(fixedPriceDF.summary())

// COMMAND ----------

// MAGIC %md
// MAGIC ## Nulls
// MAGIC 
// MAGIC There are a lot of different ways to handle null values. Sometimes, null can actually be a key indicator of the thing you are trying to predict (e.g. if you don't fill in certain portions of a form, probability of it getting approved decreases).
// MAGIC 
// MAGIC Some ways to handle nulls:
// MAGIC * Drop any records that contain nulls
// MAGIC * Numeric:
// MAGIC   * Replace them with mean/median/zero/etc.
// MAGIC * Categorical:
// MAGIC   * Replace them with the mode
// MAGIC   * Create a special category for null

// COMMAND ----------

// MAGIC %md
// MAGIC There are a few nulls in the categorical feature `host_is_superhost`. Let's get rid of those rows where any of these columns is null.

// COMMAND ----------

val noNullsDF = fixedPriceDF.na.drop(cols = Seq("host_is_superhost"))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Impute: Cast to Double
// MAGIC 
// MAGIC We requiere all numeric fields to be of type double. Let's cast all integer fields to double.

// COMMAND ----------

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

val integerColumns = for (x <- baseDF.schema.fields if (x.dataType == IntegerType)) yield x.name  
var doublesDF = noNullsDF

for (c <- integerColumns)
  doublesDF = doublesDF.withColumn(c, col(c).cast("double"))

val columns = integerColumns.mkString("\n - ")
println(s"Columns converted from Integer to Double:\n - $columns \n")
println("*-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC Add in dummy variable if we will impute any value.

// COMMAND ----------

import org.apache.spark.sql.functions.when

val imputeCols = Array(
  "bedrooms",
  "bathrooms",
  "beds", 
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value"
)

for (c <- imputeCols)
  doublesDF = doublesDF.withColumn(c + "_na", when(col(c).isNull, 1.0).otherwise(0.0))

// COMMAND ----------

display(doublesDF.describe())

// COMMAND ----------

import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer()
  .setStrategy("median")
  .setInputCols(imputeCols)
  .setOutputCols(imputeCols)

val imputedDF = imputer.fit(doublesDF).transform(doublesDF)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Getting rid of extreme values
// MAGIC 
// MAGIC Let's take a look at the *min* and *max* values of the `price` column:

// COMMAND ----------

display(imputedDF.select("price").describe())

// COMMAND ----------

// MAGIC %md
// MAGIC There are some super-expensive listings. But that's the Data Scientist's job to decide what to do with them. We can certainly filter the "free" Airbnbs though.
// MAGIC 
// MAGIC Let's see first how many listings we can find where the *price* is zero.

// COMMAND ----------

imputedDF.filter($"price" === 0).count

// COMMAND ----------

// MAGIC %md
// MAGIC Now only keep rows with a strictly positive *price*.

// COMMAND ----------

val posPricesDF = imputedDF.filter($"price" > 0)

// COMMAND ----------

// MAGIC %md
// MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

// COMMAND ----------

display(posPricesDF.select("minimum_nights").describe())

// COMMAND ----------

display(posPricesDF
  .groupBy("minimum_nights").count()
  .orderBy($"count".desc, $"minimum_nights")
)

// COMMAND ----------

// MAGIC %md
// MAGIC A minimum stay of one year seems to be a reasonable limit here. Let's filter out those records where the *minimum_nights* is greater then 365:

// COMMAND ----------

val cleanDF = posPricesDF.filter($"minimum_nights" <= 365)

display(cleanDF)

// COMMAND ----------

// MAGIC %md
// MAGIC OK, our data is cleansed now. Let's save this DataFrame to a file so that we can start building models with it.

// COMMAND ----------

val outputPath = "/tmp/sf-airbnb/sf-airbnb-clean.parquet"

cleanDF.write.mode("overwrite").parquet(outputPath)

// COMMAND ----------

