// Databricks notebook source
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.graphframes._

// COMMAND ----------

// MAGIC %md 
// MAGIC 
// MAGIC #### Basic operations with GraphFrames
// MAGIC You can create GraphFrames from vertex and edge DataFrames.
// MAGIC 
// MAGIC Vertex DataFrame: A vertex DataFrame should contain a special column named id which specifies unique IDs for each vertex in the graph.
// MAGIC Edge DataFrame: An edge DataFrame should contain two special columns: src (source vertex ID of edge) and dst (destination vertex ID of edge).
// MAGIC Both DataFrames can have arbitrary other columns. Those columns can represent vertex and edge attributes.

// COMMAND ----------

// Vertex DataFrame
val v = spark.createDataFrame(List(
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)
)).toDF("id", "name", "age")

// Edge DataFrame
val e = spark.createDataFrame(List(
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
  ("f", "c", "follow"),
  ("e", "f", "follow"),
  ("e", "d", "friend"),
  ("d", "a", "friend"),
  ("a", "e", "friend")
)).toDF("src", "dst", "relationship")

// COMMAND ----------

var g = GraphFrame(v, e)

// COMMAND ----------

display(g.vertices)

// COMMAND ----------

display(g.edges)

// COMMAND ----------

display(g.inDegrees)

// COMMAND ----------

display(g.outDegrees)

// COMMAND ----------

// MAGIC %md You can run queries directly on the vertices DataFrame. For example, we can find the age of the youngest person in the graph:

// COMMAND ----------

val youngest = g.vertices.groupBy().min("age")
display(youngest)

// COMMAND ----------

// MAGIC %md Likewise, you can run queries on the edges DataFrame. For example, let us count the number of ‘follow’ relationships in the graph:

// COMMAND ----------

val numFollows = g.edges.filter("relationship = 'follow'").count()

println(numFollows)

// COMMAND ----------

// MAGIC %md #### Motif finding
// MAGIC 
// MAGIC Build more complex relationships involving edges and vertices using motifs. The following cell finds the pairs of vertices with edges in both directions between them. The result is a DataFrame, in which the column names are motif keys.

// COMMAND ----------

// Search for pairs of vertices with edges in both directions between them.

val motifs = g.find("(a)-[e]->(b); (b)-[e2]->(a)")

display(motifs)

// COMMAND ----------

// MAGIC %md
// MAGIC #### DSL for expressing structural patterns:
// MAGIC 
// MAGIC The basic unit of a pattern is an edge. For example, "(a)-[e]->(b)" expresses an edge e from vertex a to vertex b. Note that vertices are denoted by parentheses (a), while edges are denoted by square brackets [e].
// MAGIC 
// MAGIC A pattern is expressed as a union of edges. Edge patterns can be joined with semicolons. Motif "(a)-[e]->(b); (b)-[e2]->(c)" specifies two edges from a to b to c.
// MAGIC 
// MAGIC Within a pattern, names can be assigned to vertices and edges. For example, "(a)-[e]->(b)" has three named elements: vertices a,b and edge e. These names serve two purposes:
// MAGIC 
// MAGIC 1. The names can identify common elements among edges. For example, "(a)-[e]->(b); (b)-[e2]->(c)" specifies that the same vertex b is the destination of edge e and source of edge e2.
// MAGIC 2. The names are used as column names in the result DataFrame. If a motif contains named vertex a, then the result DataFrame will contain a column “a” which is a StructType with sub-fields equivalent to the schema (columns) of GraphFrame.vertices. Similarly, an edge e in a motif will produce a column “e” in the result DataFrame with sub-fields equivalent to the schema (columns) of GraphFrame.edges.
// MAGIC 3. Be aware that names do not identify distinct elements: two elements with different names may refer to the same graph element. For example, in the motif "(a)-[e]->(b); (b)-[e2]->(c)", the names a and c could refer to the same vertex. To restrict named elements to be distinct vertices or edges, use post-hoc filters such as resultDataframe.filter("a.id != c.id").
// MAGIC 4. It is acceptable to omit names for vertices or edges in motifs when not needed. E.g., "(a)-[]->(b)" expresses an edge between vertices a,b but does not assign a name to the edge. There will be no column for the anonymous edge in the result DataFrame. Similarly, "(a)-[e]->()" indicates an out-edge of vertex a but does not name the destination vertex. These are called anonymous vertices and edges.
// MAGIC 5. An edge can be negated to indicate that the edge should not be present in the graph. E.g., "(a)-[]->(b); !(b)-[]->(a)" finds edges from a to b for which there is no edge from b to a.
// MAGIC 
// MAGIC #### Restrictions:
// MAGIC 
// MAGIC 1. Motifs are not allowed to contain edges without any named elements: "()-[]->()" and "!()-[]->()" are prohibited terms.
// MAGIC 2. Motifs are not allowed to contain named edges within negated terms (since these named edges would never appear within results). E.g., "!(a)-[ab]->(b)" is invalid, but "!(a)-[]->(b)" is valid.
// MAGIC 
// MAGIC More complex queries, such as queries which operate on vertex or edge attributes, can be expressed by applying filters to the result DataFrame.
// MAGIC 
// MAGIC This can return duplicate rows. E.g., a query "(u)-[]->()" will return a result for each matching edge, even if those edges share the same vertex u.

// COMMAND ----------

// MAGIC %md  Let us find all the reciprocal relationships in which one person is older than 30:

// COMMAND ----------

val filtered = motifs.filter("b.age > 30")
display(filtered)

// COMMAND ----------

// MAGIC %md #### Subgraphs
// MAGIC 
// MAGIC GraphFrames provides APIs for building subgraphs by filtering on edges and vertices. These filters can composed together. For example, the following subgraph contains only people who are friends and who are more than 30 years old.

// COMMAND ----------

val g2 = g
  .filterEdges("relationship = 'friend'")
  .filterVertices("age > 30")
  .dropIsolatedVertices()

// COMMAND ----------

// MAGIC %md #### Complex triplet filters
// MAGIC 
// MAGIC The following example shows how to select a subgraph based upon triplet filters that operate on an edge and its “src” and “dst” vertices. Extending this example to go beyond triplets by using more complex motifs is simple.

// COMMAND ----------

// Select subgraph based on edges "e" of type "follow"
// pointing from a younger user "a" to an older user "b".
val paths = g.find("(a)-[e]->(b)")
  .filter("e.relationship = 'follow'")
  .filter("a.age < b.age")

// COMMAND ----------

// "paths" contains vertex info. Extract the edges.
val e2 = paths.select("e.src", "e.dst", "e.relationship")
display(e2)

// We can simplify the last call as:
//  val e2 = paths.select("e.*")

// COMMAND ----------

// Construct the subgraph
val g2 = GraphFrame(g.vertices, e2)

// COMMAND ----------

display(g2.vertices)

// COMMAND ----------

display(g2.edges)

// COMMAND ----------

// MAGIC %md #### Breadth-first search (BFS)
// MAGIC 
// MAGIC Breadth-first search (BFS) finds the shortest path(s) from one vertex (or a set of vertices) to another vertex (or a set of vertices). The beginning and end vertices are specified as Spark DataFrame expressions.

// COMMAND ----------

// Search from "Esther" for users of age < 32
var paths: DataFrame = g.bfs.fromExpr("name = 'Esther'")
                            .toExpr("age < 32")
                            .run()

display(paths)

// COMMAND ----------

// The search may also limit edge filters and maximum path lengths.
val filteredPaths = g.bfs.fromExpr("name = 'Esther'")
                         .toExpr("age < 32")
                         .edgeFilter("relationship != 'friend'")
                         .maxPathLength(3)
                         .run()

display(filteredPaths)

// COMMAND ----------

// MAGIC %md #### Label propagation
// MAGIC 
// MAGIC Run static Label Propagation Algorithm for detecting communities in networks.
// MAGIC 
// MAGIC Each node in the network is initially assigned to its own community. At every superstep, nodes send their community affiliation to all neighbors and update their state to the mode community affiliation of incoming messages.
// MAGIC 
// MAGIC LPA is a standard community detection algorithm for graphs. It is inexpensive computationally, although (1) convergence is not guaranteed and (2) one can end up with trivial solutions (all nodes identify into a single community).

// COMMAND ----------

val result = g.labelPropagation.maxIter(5).run()

display(
  result.orderBy("label")
       )

// COMMAND ----------

// MAGIC %md #### PageRank
// MAGIC Identify important vertices in a graph based on connections.

// COMMAND ----------

// Run PageRank until convergence to tolerance "tol". It can also be executed for a fixed number of iterations using .maxIter(x) instead of .tol(0.01).
val results = g.pageRank
               .resetProbability(0.15)
               .tol(0.01).run()

display(results.vertices)

// COMMAND ----------

display(results.edges)

// COMMAND ----------

// Run PageRank personalized for vertex "a"
val results3 = g.pageRank
                .resetProbability(0.15)
                .maxIter(10)
                .sourceId("a").run()

display(results3.vertices)

// COMMAND ----------

// MAGIC %md ####Shortest paths
// MAGIC 
// MAGIC Computes shortest paths to the given set of landmark vertices, where landmarks specify by vertex ID.

// COMMAND ----------

val paths = g.shortestPaths
             .landmarks(Seq("a", "d"))
             .run()

display(paths)

// COMMAND ----------

// MAGIC %md #### Saving and loading GraphFrames
// MAGIC 
// MAGIC Since GraphFrames are built around DataFrames, they automatically support saving and loading to and from the same set of datasources. 

// COMMAND ----------

// Save vertices and edges as Parquet to some location.
g.vertices.write.parquet("/FileStore/vertices2")
g.edges.write.parquet("/FileStore/edges2")

// Load the vertices and edges back.
val sameV = sqlContext.read.parquet("/FileStore/vertices2")
val sameE = sqlContext.read.parquet("/FileStore/edges2")

// Create an identical GraphFrame.
val sameG = GraphFrame(sameV, sameE)