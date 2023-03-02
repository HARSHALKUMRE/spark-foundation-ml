from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("readdatajson").getOrCreate()

dataframe = spark.read.json("/config/workspace/demojson1.json")

print(type(dataframe))

dataframe.printSchema()

dataframe.show()