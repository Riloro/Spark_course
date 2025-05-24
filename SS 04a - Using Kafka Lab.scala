// Databricks notebook source
// MAGIC %md
// MAGIC # Ricardo López R. A01066515  
// MAGIC #### Notebook probado con DBR 6.4

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC <img src="https://files.training.databricks.com/images/Apache-Spark-Logo_TM_200px.png" style="float: left: margin: 20px"/>
// MAGIC 
// MAGIC # Structured Streaming with Kafka Lab
// MAGIC 
// MAGIC ## Instructions
// MAGIC * Insert solutions wherever it says `FILL_IN`
// MAGIC * Feel free to copy/paste code from the previous notebook, where applicable
// MAGIC * Run test cells to verify that your solution is correct
// MAGIC 
// MAGIC ## Prerequisites
// MAGIC * Web browser: **Chrome**
// MAGIC * Familiarity with Kafka
// MAGIC * A cluster configured with **8 cores** and **DBR 6.2**
// MAGIC * External Services
// MAGIC   - Familiarity with Kafka is helpful, but not required
// MAGIC * Suggested Courses from <a href="https://academy.databricks.com/" target="_blank">Databricks Academy</a>:
// MAGIC   - ETL Part 1
// MAGIC   - Spark-SQL

// COMMAND ----------

// MAGIC %md
// MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
// MAGIC 
// MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
// MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

// COMMAND ----------

// MAGIC %run "../Includes/Classroom-Setup"

// COMMAND ----------

// MAGIC %md
// MAGIC Define the name of the stream we are to use later in this lesson:

// COMMAND ----------

val myStreamName = "lab04a_ss"

// COMMAND ----------

// MAGIC %md
// MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Exercise 1: Use Kafka to read a stream</h2>
// MAGIC 
// MAGIC In this example, we are looking at a series of `ERROR`, `WARNING` and `INFO` log messages that are coming in via our Kafka server.
// MAGIC 
// MAGIC We want to analyze how many log messages are coming from each IP address?
// MAGIC 
// MAGIC Create `initialDF` with the following Kafka parameters:
// MAGIC 
// MAGIC 0. `format` is `kafka`
// MAGIC 0. `kafka.bootstrap.server` (pick the server closest to you)
// MAGIC   * is `server1.databricks.training:9092` US (Oregon)
// MAGIC   * or `server2.databricks.training:9092` Singapore
// MAGIC 0. `subscribe` is `logdata`
// MAGIC 
// MAGIC When you are done, run the TEST cell that follows to verify your results.

// COMMAND ----------

// TODO
spark.conf.set("spark.sql.shuffle.partitions", sc.defaultParallelism)

val kafkaServer = "server1.databricks.training:9092" // Specify "kafka" bootstrap server

// Create our initial DataFrame
val initialDF = spark.readStream
 .format("kafka")              // Specify "kafka" as the type of the stream
 .option("kafka.bootstrap.servers", kafkaServer)              // Set the location of the kafka server
 .option("subscribe", "logdata")              // Indicate which topics to listen to
 .option("startingOffsets", "earliest")       // Rewind stream to beginning when we restart notebook
 .option("maxOffsetsPerTrigger", 1000)              // Throttle Kafka's processing of the streams
 .load()             // Load the input data stream in as a DataFrame

// COMMAND ----------

// TEST - Run this cell to test your solution.
lazy val initSchemaStr = initialDF.schema.mkString("")

dbTest("SS-04-key",       true, initSchemaStr.contains("(key,BinaryType,true)"))
dbTest("SS-04-value",     true, initSchemaStr.contains("(value,BinaryType,true)"))
dbTest("SS-04-topic",     true, initSchemaStr.contains("(topic,StringType,true)"))
dbTest("SS-04-partition", true, initSchemaStr.contains("(partition,IntegerType,true)"))
dbTest("SS-04-offset",    true, initSchemaStr.contains("(offset,LongType,true)"))
dbTest("SS-04-timestamp", true, initSchemaStr.contains("(timestamp,TimestampType,true)"))
dbTest("SS-04-timestampType", true, initSchemaStr.contains("(timestampType,IntegerType,true)"))

println("Tests passed!")
println("-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Exercise 2: Do Some ETL Processing</h2>
// MAGIC 
// MAGIC Perform the following ETL steps:
// MAGIC 
// MAGIC 0. Cast `value` column to STRING
// MAGIC 0. `ts_string` is derived from `value` at positions 14 to 24,
// MAGIC 0. `epoc` is derived from `unix_timestamp` of `ts_string` using format "yyyy/MM/dd HH:mm:ss.SSS"
// MAGIC 0. `capturedAt` is derived from casting `epoc` to `timestamp` format
// MAGIC 0. `logData` is created by applying `regexp_extract` on `value`.. use this string `"""^.*\]\s+(.*)$"""`
// MAGIC 
// MAGIC When you are done, run the TEST cell that follows to verify your results.

// COMMAND ----------

// TODO
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
var cleanDF = initialDF
 .withColumn("value", $"value".cast("STRING"))  // Select the "value" column, cast "value" column to STRING
 .withColumn("ts_string",substring(col("value"),14,24) )  // Select the "value" column, pull substring(14,24) from it and rename to "ts_string"
 .withColumn("epoc", unix_timestamp($"ts_string", "yyyy/MM/dd HH:mm:ss.SSS" ))  // Select the "ts_string" column, apply unix_timestamp to it and rename to "epoc"
 .withColumn("capturedAt", to_timestamp($"epoc"))  // Select the "epoc" column and cast to a timestamp and rename it to "capturedAt"
 .withColumn("logData", regexp_extract($"value", """^.*\]\s+(.*)$""" ,1))  // Select the "logData" column and apply the regexp `"""^.*\]\s+(.*)$"""`

// COMMAND ----------

// TEST - Run this cell to test your solution.
lazy val schemaStr = cleanDF.schema.mkString("")

dbTest("SS-04-schema-value",     true, schemaStr.contains("(value,StringType,true)"))
dbTest("SS-04-schema-ts_string",  true, schemaStr.contains("(ts_string,StringType,true)"))
dbTest("SS-04-schema-epoc",   true, schemaStr.contains("(epoc,LongType,true)"))
dbTest("SS-04-schema-capturedAt", true, schemaStr.contains("(capturedAt,TimestampType,true"))
dbTest("SS-04-schema-logData",  true, schemaStr.contains("(logData,StringType,true)"))

println("Tests passed!")
println("-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Exercise 3: Classify and Count IP Addresses Over a 10s Window</h2>
// MAGIC 
// MAGIC To solve this problem, you need to:
// MAGIC 
// MAGIC 0. Parse the first part of an IP address from the column `logData` with the `regexp_extract()` function
// MAGIC   * You will need a regular expression which we have already provided below as `IP_REG_EX`
// MAGIC   * Hint: take the 1st matching value
// MAGIC 0. Filter out the records that don't contain IP addresses
// MAGIC 0. Form another column called `ipClass` that classifies IP addresses based on the first part of an IP address 
// MAGIC   * 1 to 126: "Class A"
// MAGIC   * 127: "Loopback"
// MAGIC   * 128 to 191: "Class B"
// MAGIC   * 192 to 223: "Class C"
// MAGIC   * 224 to 239: "Class D"
// MAGIC   * 240 to 256: "Class E"
// MAGIC   * anything else is invalid
// MAGIC 0. Perform an aggregation over a window of time, grouping by the `capturedAt` window and `ipClass`
// MAGIC   * For this lab, use a 10-second window
// MAGIC 0. Count the number of IP values that belong to a specific `ipClass`
// MAGIC 0. Sort by `ipClass`

// COMMAND ----------

// TODO
import org.apache.spark.sql.functions.{length, window, when}

//This is the regular expression pattern that we will use 
val IP_REG_EX = """^.*\s+(\d{1,3})\.\d{1,3}\.\d{1,3}\.\d{1,3}.*$"""

val ipDF = cleanDF
 .withColumn("ip", regexp_extract($"logData",IP_REG_EX  ,1))  
 .withColumn("ip", $"ip".cast("int"))// apply regexp_extract on IP_REG_EX with value of 1 to "logData" and rename it "ip"
 .filter(length($"ip")> 0)                          // keep only "ip" that have non-zero length
 .withColumn("ipClass", when( $"ip" >= 1 && $"ip" <= 126, "Class A")// figure out class of IP address based on first two octets
                       .when($"ip" === 127,"Loopback")
                       .when($"ip" >= 128 && $"ip" <= 191, "Class B")
                       .when($"ip" >= 192 && $"ip" <= 223, "Class C")
                       .when($"ip" >= 224 && $"ip" <= 239, "Class D")
                       .when($"ip" >= 240 && $"ip" <= 256, "Class E")// add rest of when/otherwise clauses
                       .otherwise("invalid")
            )
 .groupBy($"ipClass", window($"capturedAt", "10 seconds").as("time"))                // form 10 second windows of "capturedAt", call them "time" 
 .count()                                           // add up total
 .orderBy("ipClass")      

// COMMAND ----------

// TEST     - Run this cell to test your solution.
lazy val schemaStr = ipDF.schema.mkString("")

dbTest("SS-04-schema-ipClass", true, schemaStr.contains("(ipClass,StringType,false)"))
dbTest("SS-04-schema-count",   true, schemaStr.contains("(count,LongType,false)"))
dbTest("SS-04-schema-start",   true, schemaStr.contains("(start,TimestampType,true)"))
dbTest("SS-04-schema-end",     true, schemaStr.contains("(end,TimestampType,true)"))

println("Tests passed!")
println("-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Exercise 4: Display a LIVE Plot</h2>
// MAGIC 
// MAGIC The `DataFrame` that you pass to `display()` should have three columns:
// MAGIC 
// MAGIC * `time`: The time window structure
// MAGIC * `ipClass`: The class the first part of the IP address belongs to
// MAGIC * `count`: The number of times that said class of IP address appeared in the window
// MAGIC 
// MAGIC Under <b>Plot Options</b>, use the following:
// MAGIC * <b>Keys:</b> `ipClass`
// MAGIC * <b>Values:</b> `count`
// MAGIC 
// MAGIC <b>Display type:</b> is 
// MAGIC * <b>Pie Chart</b>
// MAGIC 
// MAGIC <img src="https://files.training.databricks.com/images/eLearning/Structured-Streaming/plot-options-pie.png"/>

// COMMAND ----------

// TODO
display(ipDF, streamName = myStreamName)

// COMMAND ----------

// TEST - Run this cell to test your solution.
dbTest("SS-04-numActiveStreams", true, spark.streams.active.length > 0)

println("Tests passed!")

// COMMAND ----------

// MAGIC %md
// MAGIC Wait until stream is done initializing...

// COMMAND ----------

untilStreamIsReady(myStreamName)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> Exercise 5: Stop streaming jobs</h2>
// MAGIC 
// MAGIC Before we can conclude, we need to shut down all active streams.

// COMMAND ----------

// TODO
 
for (stream <- spark.streams.active) {             // Iterate over all active streams
  println("stopping " + stream.name)  // A little console output
  try {

    stream.stop()  // Stop the stream

  } catch {
    case e:Exception => e.printStackTrace()
  }                            
}

// COMMAND ----------

// TEST - Run this cell to test your solution.
dbTest("SS-04-numActiveStreams", 0, spark.streams.active.length)

println("Tests passed!")

// COMMAND ----------

// MAGIC %md
// MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
// MAGIC 
// MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

// COMMAND ----------

// MAGIC %run "../Includes/Classroom-Cleanup"

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>
// MAGIC 
// MAGIC Thank you for your participation!

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
// MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
// MAGIC <br/>
// MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>