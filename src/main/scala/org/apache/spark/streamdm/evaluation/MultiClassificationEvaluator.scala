package org.apache.spark.streamdm.evaluation

import org.apache.spark.streamdm.core.Example
import org.apache.spark.streaming.dstream.DStream

/**
  * Created by Robert on 13.01.2017.
  */
class MultiClassificationEvaluator extends Evaluator{

  var numInstancesCorrect = 0
  var numInstancesSeen = 0

  /**
    * Process the result of a predicted stream of Examples and Doubles.
    *
    * @param input the input stream containing (Example,Double) tuples
    * @return a stream of String with the processed evaluation
    */
  override def addResult(input: DStream[(Example, Double)]): DStream[String] = {
    input.map(x=>isCorrect(x)).reduce((x,y)=>(x._1+y._1,x._2+y._2))
        .map(x => {"%.4f"
      .format(x._1/(x._1+x._2))})
  }

  def isCorrect(x: (Example, Double)):(Double, Double) = {
    if(x._1.labelAt(0)==x._2) (1.0,0.0) else (0.0,1.0)
  }

  /**
    * Get the evaluation result.
    *
    * @return a Double containing the evaluation result
    */
  override def getResult(): Double =
    numInstancesCorrect.toDouble/numInstancesSeen.toDouble
}
