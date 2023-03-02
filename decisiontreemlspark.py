from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

spark = SparkSession\
        .builder\
        .appName("DecisionTreeWithSpark")\
        .getOrCreate()

dataset = spark.read.csv("/config/workspace/winequality_red.csv", header=True)

#dataset.show()

#dataset.printSchema()

# spark type casting function

from pyspark.sql.functions import col
new_dataset = dataset.select(*(col(c).cast("float").alias(c) for c in dataset.columns))

new_dataset.printSchema()