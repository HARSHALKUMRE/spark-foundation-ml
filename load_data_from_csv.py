from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('readdatacsv').getOrCreate()

dataframe = spark.read.csv("/config/workspace/file.csv")

print(type(dataframe))

dataframe.printSchema()

dataframe.show()
