import core.NeuralNetwork
import data.{IrisDataSetLoader, DataSet}
import training.Trainer

import scala.language.postfixOps

object Application {

    // iris dataset
    def main(args: Array[String]) {
        val numFeatures = 4
        val numClasses = 3
        val hiddenNodes = 10
        val numberOfEpochs = 200
        val trainingSplitRatio = 0.6


        val irisDataSet: DataSet = IrisDataSetLoader.load(trainingSplitRatio)

        val net = NeuralNetwork(numFeatures, hiddenNodes, numClasses)

        Trainer.train(net, irisDataSet, epochs = numberOfEpochs)

    }
}
