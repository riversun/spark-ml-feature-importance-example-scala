/*
 *  Copyright (c) 2019 Tom Misawa, riversun.org@gmail.com
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *
 */
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.SparkSession
import org.riversun.ml.spark.FeatureImportance
import org.riversun.ml.spark.FeatureImportance.Order

object FeatureImportancesExample {

  def main(args: Array[String]): Unit = {

    org.apache.log4j.Logger.getLogger("org").setLevel(org.apache.log4j.Level.ERROR)
    org.apache.log4j.Logger.getLogger("akka").setLevel(org.apache.log4j.Level.ERROR)

    System.setProperty("hadoop.home.dir", "c:\\Temp\\winutil\\") //for windows

    val spark = SparkSession
      .builder
      .appName("GradientBoostedTreeRegressorExample")
      .master("local[*]")
      .getOrCreate()

    val dataset = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data/mllib/gem_price.csv") // gem_price_ja.csv for Japanese

    val stringIndexers = Array("material", "shape", "brand", "shop")
      .map { colName =>
        new StringIndexer()
          .setInputCol(colName)
          .setOutputCol(colName + "Index")
      }

    val assembler = new VectorAssembler()
      .setInputCols(stringIndexers.map(indexer => indexer.getOutputCol) :+ "weight")
      .setOutputCol("features")

    val gbtr = new GBTRegressor()
      .setLabelCol("price")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(stringIndexers :+ assembler :+ gbtr);

    val splits = dataset.randomSplit(Array(0.7, 0.3), 1L)
    val trainingData = splits(0)
    val testData = splits(1)

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    val gbtModel = model.stages.last.asInstanceOf[GBTRegressionModel];
    val schema = predictions.schema

    val importances = new FeatureImportance.Builder(gbtModel, schema)
      .sort(Order.DESCENDING)
      .build.getResult

    importances.forEach(println)

    spark.stop()
  }
}