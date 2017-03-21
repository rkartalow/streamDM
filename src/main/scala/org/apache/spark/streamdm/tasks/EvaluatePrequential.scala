/*
 * Copyright (C) 2015 Holmes Team at HUAWEI Noah's Ark Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.apache.spark.streamdm.tasks

import com.github.javacliparser.ClassOption
import org.apache.spark.streamdm.classifiers._
import org.apache.spark.streamdm.evaluation.Evaluator
import org.apache.spark.streamdm.streams._
import org.apache.spark.streaming.StreamingContext

/**
  * Task for evaluating a classifier on a stream by testing then training with
  * each example in sequence.
  *
  * <p>It uses the following options:
  * <ul>
  * <li> Learner (<b>-c</b>), an object of type <tt>Classifier</tt>
  * <li> Evaluator (<b>-e</b>), an object of type <tt>Evaluator</tt>
  * <li> Reader (<b>-s</b>), a reader object of type <tt>StreamReader</tt>
  * <li> Writer (<b>-w</b>), a writer object of type <tt>StreamWriter</tt>
  * </ul>
  */
class EvaluatePrequential extends Task {

  val decor = 1 - (Math.log(41) / Math.log(2) + 1) / 41

  val classifier = "trees.RandomForestShuffle" +
    //   " -j 50000" + //maximum number of observations per tree (RandomForestSequential only)
    " -i 20" + //number of trees (RandomForestModels only)
    " -d " + decor + //decorrelation (RandomForestModels only)
    " -o" + //grow
    " -b" + //binary splits only
    " -p" //no pre-pruning
  //  " -g 100" + //min obs between splits
  //  " -c 0.05" //split conf

  val learnerOption: ClassOption = new ClassOption("learner", 'l',
    "Learner to use", classOf[Classifier], classifier)

  val evaluatorOption:ClassOption = new ClassOption("evaluator", 'e',
    "Evaluator to use", classOf[Evaluator], "MultiClassificationEvaluator")

  val streamReaderOption:ClassOption = new ClassOption("streamReader", 's',
    "Stream reader to use", classOf[StreamReader], "FileReader -f C://Users//robert.kartalow//IdeaProjects//streamDM//data//KDDCUP99_full_s.arff -k 10000")

  val resultsWriterOption:ClassOption = new ClassOption("resultsWriter", 'w',
    "Stream writer to use", classOf[StreamWriter], "PrintStreamWriter")

  /**
    * Run the task.
    *
    * @param ssc The Spark Streaming context in which the task is run.
    */
  def run(ssc:StreamingContext): Unit = {

    val reader:StreamReader = this.streamReaderOption.getValue()

    val learner:Classifier = this.learnerOption.getValue()
    learner.init(reader.getExampleSpecification())

    val evaluator:Evaluator = this.evaluatorOption.getValue()

    val writer:StreamWriter = this.resultsWriterOption.getValue()

    val instances = reader.getExamples(ssc)

    System.out.println(classifier)

    //Predict
    val predPairs = learner.predict(instances)

    //Evaluate
    writer.output(evaluator.addResult(predPairs))

    //Train
    learner.train(instances)

  }
}