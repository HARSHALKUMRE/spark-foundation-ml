from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('readdatacsv').getOrCreate()

dataframe = spark.read.csv("/config/workspace/winequality_red.csv")

print(type(dataframe))

dataframe.printSchema()

dataframe.show()
