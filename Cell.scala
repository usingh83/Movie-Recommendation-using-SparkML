package lab

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.Rating
object Cells {
  def main(args: Array[String])
  {
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  val sc= new SparkContext("local[*]", "Ques2")
  val rate = sc.textFile(args(0))
  val rating = rate.map(_.split("::") match { case Array(user, item, rate, timestamp) => (timestamp.toLong%10,Rating(user.toInt, item.toInt, rate.toDouble))})
  val numPartitions = 4
  val training = rating.filter(x => x._1 < 6).values.repartition(numPartitions).cache()
  val test = rating.filter(x => x._1 >= 6).values.repartition(numPartitions).cache()
  var rank = 10
  var lambda = 0.6
  var numIter = 20
  val model = ALS.train(training, rank, numIter, lambda)
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }
  val rmse = computeRmse(model, test, test.count())
  println("RMSE (test) = " + rmse + " for the model trained with rank = " + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")  
  }
 }
                  