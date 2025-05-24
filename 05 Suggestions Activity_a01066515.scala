// Databricks notebook source
// MAGIC %md ### In-class activity
// MAGIC 
// MAGIC #### Increasing the suggested boxes to the customers.
// MAGIC 
// MAGIC Your beverage company makes suggestions to many customers about how many product boxes should they buy. The data science team has a machine learning model that outputs a continuous number of the optimal suggestion per customer-sku. It is your task to transform that suggestion into a real one by converting it in complete boxes (a customer cannot buy 1.5 boxes, only 1 or 2). 
// MAGIC 
// MAGIC The suggestions look like this:
// MAGIC 
// MAGIC | Customer | SKU_group | SKU | suggestion |
// MAGIC |:--------:|:---------:|:---:|:----------:|
// MAGIC |    500   | Refrescos | 100 |    10.7    |
// MAGIC |    500   | Refrescos | 101 |     2.6    |
// MAGIC |    500   |   Leche   | 243 |     1.9    |
// MAGIC |    500   |   Leche   | 244 |     1.5    |
// MAGIC |    500   |   Leche   | 298 |     2.2    |
// MAGIC 
// MAGIC The customer should receive an integer suggestion; however, the remainders are important. In case that the remainders per sku_group sum a box, then we assign it to the one that is closest to become the next integer. For example, in the Leche group we have the following remainders:
// MAGIC 
// MAGIC |  SKU  | remainder |
// MAGIC |:-----:|:---------:|
// MAGIC |  243  |    0.9    |
// MAGIC |  244  |    0.5    |
// MAGIC |  298  |    0.2    |
// MAGIC | TOTAL |    1.6    |
// MAGIC 
// MAGIC In total, there are 1.6 boxes that will not be suggested if we just take the integer part of the suggestion. In this case, we proceed to increase at most 1 box the product that is closest to increase from the original suggestion, which in the case of Leche is SKU 243.
// MAGIC 
// MAGIC | SKU |  To_next_box  |
// MAGIC |:---:|:-------------:|
// MAGIC | 243 | 1.9 – 2 = 0.1 |
// MAGIC | 244 | 1.5 – 2 = 0.5 |
// MAGIC | 298 | 2.2 – 3 = 0.8 |
// MAGIC 
// MAGIC In the end, the suggestion should look like the following:
// MAGIC 
// MAGIC | Customer | SKU_group | SKU | suggestion | Final_suggestion |
// MAGIC |:--------:|:---------:|:---:|:----------:|:----------------:|
// MAGIC |    500   | Refrescos | 100 |    10.7    |        11        |
// MAGIC |    500   | Refrescos | 101 |     2.6    |         2        |
// MAGIC |    500   |   Leche   | 243 |     1.9    |         2        |
// MAGIC |    500   |   Leche   | 244 |     1.5    |         1        |
// MAGIC |    500   |   Leche   | 298 |     2.2    |         2        |

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window

// COMMAND ----------

// Lectura de datos ... 
var data = spark.read.format("csv").option("header", true).load("/FileStore/tables/suggestions.csv")
//Changind the data type of suggestion and Customer 
data = data.withColumn("suggestion", $"suggestion".cast("double")).withColumn("Customer", $"Customer".cast("int"))
display(data)

// COMMAND ----------

//Computing the remainder and the diffeference to the next integer for each record...
var df_2 = data.select($"*", ($"suggestion" - floor($"suggestion")).alias("remainder"),
            (ceil($"suggestion") - $"suggestion").alias("to_next_box"))
display(df_2)

// COMMAND ----------

//display(df_2.groupBy($"Customer".alias("Customer_2"),$"sku_group".alias("sku_group_2")).sum("remainder").filter($"Customer_2" === "-2143627985"))

// COMMAND ----------

//Computing the remainder sum for each customer and sku_group ...
var sum_remainder_grp = df_2.groupBy($"Customer".alias("Customer_2"),$"sku_group".alias("sku_group_2")).sum("remainder")
//Join between the dataFrame with all records and sum_remainder_grp dataFrame ...
var df_3 = df_2.join(sum_remainder_grp, df_2("Customer") === sum_remainder_grp("Customer_2") && 
         df_2("sku_group") === sum_remainder_grp("sku_group_2"), "inner")
var df_4 = df_3.drop("Customer_2","sku_group_2").filter($"suggestion">0)
display(df_4)

// COMMAND ----------

// MAGIC %md
// MAGIC # Resultado final:

// COMMAND ----------

//Grouping rows by a window ...
val w = Window.partitionBy("Customer", "sku_group")
// If the dataFrame is already groupBy customer and sku_group, then
//we use the rank to give a  place to each value in  to_next_box
var df_new =  df_4.withColumn("rank_to_next_box", row_number() over Window.partitionBy("Customer","sku_group").orderBy($"to_next_box".asc))    
                             .withColumn("final_suggestion", when( $"rank_to_next_box" <= floor($"sum(remainder)"), ceil($"suggestion") ) 
                             .otherwise(floor($"suggestion")))
//Displaying the FINAL RESULT of suggestions ... 
display(df_new.select($"Customer",$"sku_group",$"sku",$"suggestion",$"final_suggestion")
              .orderBy($"customer".desc, $"sku_group"))

// COMMAND ----------

////////////////////////////////////////////CELDA DE PRUEBAS //////////////////////////////////////////////////////////////////////////////
// If the dataFrame is already groupBy customer and sku_group, then
var test =  df_4.filter($"Customer" === -2141205365 && $"sku_group" === "SABORES REGULAR FAMILIAR NO RETORNABLE")
                .filter($"suggestion" > 0) //Filter rows with suggestion > 0
                .withColumn("rank_to_next_box", row_number() over Window.partitionBy().orderBy($"to_next_box".asc)) //we use the rank to give a  place to each value in to_next_box
                .withColumn("final_suggestion", when( $"rank_to_next_box" <=  floor($"sum(remainder)"), ceil($"suggestion") )
                            .otherwise(floor($"suggestion")))
                //.withColum("final_suggestion", )
display(test)

// COMMAND ----------

