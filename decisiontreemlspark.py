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

from pyspark.sql.functions import col, count, isnan, when

new_dataset.select([count(when(col(c).isNull(), c)).alias (c) for c in new_dataset.columns]).show()

cols = new_dataset.columns

cols.remove("quality")

assembler = VectorAssembler(inputCols=cols, outputCol="features")

data = assembler.transform(new_dataset)

data = data.select("features", "quality")

data.show()

from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol="quality", outputCol="quality_index")

data_Indexed = stringIndexer.fit(data).transform(data)

data_Indexed.show()

# split the data
(train, test) = data_Indexed.randomSplit([0.7, 0.3])

# module building
dt = DecisionTreeClassifier(labelCol="quality_index", featuresCol="features")
model = dt.fit(train)

# for prediction
prediction = model.transform(test)

# for showcase the prediction
prediction.show()

# Evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="quality_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction)
print("Accuracy:- ",accuracy)

# Saving the model
model.save("decisiontreeclassifier.pkl")

# load the model
model.load("")

# stop spark
# spark.stop()