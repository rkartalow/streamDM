package org.apache.spark.streamdm.classifiers.trees

import com.github.javacliparser.{ClassOption, FlagOption, FloatOption, IntOption}
import org.apache.spark.Logging
import org.apache.spark.streamdm.classifiers.Classifier
import org.apache.spark.streamdm.core.specification.ExampleSpecification
import org.apache.spark.streamdm.core.{Example, Model}
import org.apache.spark.streaming.dstream.DStream

import scala.collection.parallel.mutable.ParArray
import scala.util.Random

/**
  * Created by Robert on 14.01.2017.
  */
class RandomForest extends Classifier {

  type T = RandomForestModel

  val numTreesOption: IntOption = new IntOption("numTrees", 'i',
    "Number of trees to grow", 10, 1, Int.MaxValue)

  val decorrelationOption: FloatOption = new FloatOption("decorrelation", 'd',
    "Ratio of features that should randomly disabled for each tree", 0.2, 0.0, 0.9999)

  val numericObserverTypeOption: IntOption = new IntOption("numericObserverType", 'n',
    "numeric observer type, 0: gaussian", 0, 0, 2)

  val splitCriterionOption: ClassOption = new ClassOption("splitCriterion", 's',
    "Split criterion to use.", classOf[SplitCriterion], "InfoGainSplitCriterion")

  val growthAllowedOption: FlagOption = new FlagOption("growthAllowed", 'o',
    "Allow to grow")

  val binaryOnlyOption: FlagOption = new FlagOption("binaryOnly", 'b',
    "Only allow binary splits")

  val numGraceOption: IntOption = new IntOption("numGrace", 'g',
    "The number of examples a leaf should observe between split attempts.",
    100, 1, Int.MaxValue)

  val tieThresholdOption: FloatOption = new FloatOption("tieThreshold", 't',
    "Threshold below which a split will be forced to break ties.", 0.05, 0, 1)

  val splitConfidenceOption: FloatOption = new FloatOption("splitConfidence", 'c',
    "The allowable error in split decision, values closer to 0 will take longer to decide.",
    0.05, 0.0, 1.0)

  val learningNodeOption: IntOption = new IntOption("learningNodeType", 'l',
    "Learning node type of leaf", 0, 0, 2)

  val nbThresholdOption: IntOption = new IntOption("nbThreshold", 'q',
    "The number of examples a leaf should observe between permitting Naive Bayes", 0, 0, Int.MaxValue)

  val noPrePruneOption: FlagOption = new FlagOption("noPrePrune", 'p', "Disable pre-pruning.")

  val removePoorFeaturesOption: FlagOption = new FlagOption("removePoorFeatures", 'r', "Disable poor features.")

  val splitAllOption: FlagOption = new FlagOption("SplitAll", 'a', "Split at all leaves")

  var model: RandomForestModel = _

  var espec: ExampleSpecification = _

  /* Init the model used for the Learner*/
  override def init(exampleSpecification: ExampleSpecification): Unit = {
    espec = exampleSpecification
    val numFeatures = espec.numberInputFeatures()
    val outputSpec = espec.outputFeatureSpecification(0)
    val numClasses = outputSpec.range()
    model = new RandomForestModel(espec, numTreesOption.getValue(), numericObserverTypeOption.getValue, splitCriterionOption.getValue(),
      growthAllowedOption.isSet(), binaryOnlyOption.isSet(), numGraceOption.getValue(),
      tieThresholdOption.getValue, splitConfidenceOption.getValue(),
      learningNodeOption.getValue(), nbThresholdOption.getValue(),
      noPrePruneOption.isSet(), removePoorFeaturesOption.isSet(), splitAllOption.isSet(), decorrelationOption.getValue)
    model.init()
  }

  /* Predict the label of the Example stream, given the current Model
     *
     * @param instance the input Example stream
     * @return a stream of tuples containing the original instance and the
     * predicted value
     */
  override def predict(input: DStream[Example]): DStream[(Example, Double)] = {
    input.map(x => (x,model.predict(x)))
  }

  /**
    * Train the model based on the algorithm implemented in the learner,
    * from the stream of Examples given for training.
    *
    * @param input a stream of Examples
    */
  override def train(input: DStream[Example]): Unit = {
    val numTrees = numTreesOption.getValue
    var count = 0
    input.foreachRDD {
      rdd =>
        count+=1
        System.out.println(count + ". Training start: " + System.currentTimeMillis())
        val i = new scala.util.Random().nextInt(numTrees)
        val tmodel = rdd.aggregate(
          new HoeffdingTreeModel(model.trees(i)))(
          (mod, example) => {
            mod.update(example)
          }, (mod1, mod2) => mod1.merge(mod2, trySplit = false))
        model.trees(i) = model.trees(i).merge(tmodel, trySplit = true)
        System.out.println(count + ". Training end: " + System.currentTimeMillis())
    }
  }

  /**
    * Gets the current Model used for the Learner.
    *
    * @return the Model object used for training
    */
  override def getModel: RandomForestModel = model

}

class RandomForestShuffle extends Classifier {

  type T = RandomForestModel

  val numTreesOption: IntOption = new IntOption("numTrees", 'i',
    "Number of trees to grow", 10, 1, Int.MaxValue)

  val decorrelationOption: FloatOption = new FloatOption("decorrelation", 'd',
    "Ratio of features that should randomly disabled for each tree", 0.2, 0.0, 0.9999)

  val numericObserverTypeOption: IntOption = new IntOption("numericObserverType", 'n',
    "numeric observer type, 0: gaussian", 0, 0, 2)

  val splitCriterionOption: ClassOption = new ClassOption("splitCriterion", 's',
    "Split criterion to use.", classOf[SplitCriterion], "InfoGainSplitCriterion")

  val growthAllowedOption: FlagOption = new FlagOption("growthAllowed", 'o',
    "Allow to grow")

  val binaryOnlyOption: FlagOption = new FlagOption("binaryOnly", 'b',
    "Only allow binary splits")

  val numGraceOption: IntOption = new IntOption("numGrace", 'g',
    "The number of examples a leaf should observe between split attempts.",
    100, 1, Int.MaxValue)

  val tieThresholdOption: FloatOption = new FloatOption("tieThreshold", 't',
    "Threshold below which a split will be forced to break ties.", 0.05, 0, 1)

  val splitConfidenceOption: FloatOption = new FloatOption("splitConfidence", 'c',
    "The allowable error in split decision, values closer to 0 will take longer to decide.",
    0.05, 0.0, 1.0)

  val learningNodeOption: IntOption = new IntOption("learningNodeType", 'l',
    "Learning node type of leaf", 0, 0, 2)

  val nbThresholdOption: IntOption = new IntOption("nbThreshold", 'q',
    "The number of examples a leaf should observe between permitting Naive Bayes", 0, 0, Int.MaxValue)

  val noPrePruneOption: FlagOption = new FlagOption("noPrePrune", 'p', "Disable pre-pruning.")

  val removePoorFeaturesOption: FlagOption = new FlagOption("removePoorFeatures", 'r', "Disable poor features.")

  val splitAllOption: FlagOption = new FlagOption("SplitAll", 'a', "Split at all leaves")

  var model: RandomForestModel = _

  var espec: ExampleSpecification = _

  /* Init the model used for the Learner*/
  override def init(exampleSpecification: ExampleSpecification): Unit = {
    espec = exampleSpecification
    val numFeatures = espec.numberInputFeatures()
    val outputSpec = espec.outputFeatureSpecification(0)
    val numClasses = outputSpec.range()
    model = new RandomForestModel(espec, numTreesOption.getValue(), numericObserverTypeOption.getValue, splitCriterionOption.getValue(),
      growthAllowedOption.isSet(), binaryOnlyOption.isSet(), numGraceOption.getValue(),
      tieThresholdOption.getValue, splitConfidenceOption.getValue(),
      learningNodeOption.getValue(), nbThresholdOption.getValue(),
      noPrePruneOption.isSet(), removePoorFeaturesOption.isSet(), splitAllOption.isSet(), decorrelationOption.getValue)
    model.init()
  }

  /* Predict the label of the Example stream, given the current Model
     *
     * @param instance the input Example stream
     * @return a stream of tuples containing the original instance and the
     * predicted value
     */
  override def predict(input: DStream[Example]): DStream[(Example, Double)] = {
    input.map(x => (x,model.predict(x)))
  }

  /**
    * Train the model based on the algorithm implemented in the learner,
    * from the stream of Examples given for training.
    *
    * @param input a stream of Examples
    */
  override def train(input: DStream[Example]): Unit = {
    val numTrees = numTreesOption.getValue
    var count = 0
    input.foreachRDD {
      rdd =>
        count += 1
        System.out.println(count + ". Training start: " + System.currentTimeMillis())
        var tmodels = rdd.map(example => (new scala.util.Random().nextInt(numTrees),example))
          .map(x => (x._1, (x._1, x._2)))
          .aggregateByKey(new RandomForestModel(model))(
            (mod,x)=> mod.update(x._2, x._1) , (mod1, mod2) => mod1.merge(mod2))
          .collect()
        for(x<-tmodels) {
          model.trees(x._1) = model.trees(x._1).merge(x._2.trees(x._1), trySplit = true)
        }
        System.out.println(count + ". Training end: " + System.currentTimeMillis())
    }
  }

  /**
    * Gets the current Model used for the Learner.
    *
    * @return the Model object used for training
    */
  override def getModel: RandomForestModel = model

}

class RandomForestModel(val espec: ExampleSpecification, val numTrees: Int = 50, val numericObserverType: Int = 0,
                         val splitCriterion: SplitCriterion = new InfoGainSplitCriterion(),
                         var growthAllowed: Boolean = true, val binaryOnly: Boolean = true,
                         val graceNum: Int = 200, val tieThreshold: Double = 0.05,
                         val splitConfedence: Double = 0.05, val learningNodeType: Int = 0,
                         val nbThreshold: Int = 0, val noPrePrune: Boolean = true,
                         val removePoorFeatures: Boolean = false, val splitAll: Boolean = false,
                         val decorrelation: Double = 0.5)
  extends Model with Serializable with Logging {

  type T = RandomForestModel
  var trees: ParArray[HoeffdingTreeModel] = _

  def this(model: RandomForestModel) {
    this(model.espec, model.numTrees, model.numericObserverType, model.splitCriterion, model.growthAllowed,
      model.binaryOnly, model.graceNum, model.tieThreshold, model.splitConfedence,
      model.learningNodeType, model.nbThreshold, model.noPrePrune, model.removePoorFeatures, model.splitAll,
      model.decorrelation)
    trees = model.trees
  }

  /* init the model */
  def init(): Unit = {
    trees = new ParArray[HoeffdingTreeModel](numTrees)
    var featureIndeces = 0 until espec.numberInputFeatures()
    for (i <- 0 until numTrees) {
      var disabledFeatures: Array[Int] = null
      if(decorrelation>0.0) {
        disabledFeatures = Random.shuffle(featureIndeces.toList).take((decorrelation*espec.numberInputFeatures()).toInt).toArray
      }
      trees(i) = new HoeffdingTreeModel(espec, numericObserverType, splitCriterion,
        growthAllowed, binaryOnly, graceNum,
        tieThreshold, splitConfedence,
        learningNodeType, nbThreshold,
        noPrePrune, removePoorFeatures, splitAll, disabledFeatures)
      trees(i).init()
    }
  }

  def reduceFunc(v1: (Int, Example), v2: (Int, Example)) : (Int, Example) = {
    trees(v1._1)=trees(v1._1).update(v1._2)
    v2
  }

  def predict(example: Example): Double = {
    //trees.map(x => (example, x.predict(example)))
    var classVotes = new Array[Int](espec.outputFeatureSpecification(0).range())
    for (i <- 0 until numTrees) {
      val prediction = trees(i).predict(example).toInt
      if (prediction != -1) classVotes(prediction)+=1
    }
    //System.out.println("Number of votes: "+ classVotes.sum)
    classVotes.indexOf(classVotes.max).toDouble
  }

  /**
    * Update the model, depending on the Instance given for training.
    *
    * @param change the example based on which the Model is updated
    * @return the updated Model
    */
  override def update(change: Example): RandomForestModel = this

  def update(change: Example, tree: Int): RandomForestModel = {
    trees(tree) = trees(tree).update(change)
    this
  }

  def merge(that: RandomForestModel): RandomForestModel = {
    for(i<-0 until numTrees) {
      trees(i) = this.trees(i).merge(that.trees(i), trySplit = false)
    }
    this
  }

  def replaceTree(tree: HoeffdingTreeModel, treeIndex: Int): RandomForestModel = {
    trees(treeIndex) = tree
    this
  }

}