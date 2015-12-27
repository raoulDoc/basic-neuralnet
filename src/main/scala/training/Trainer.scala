package training

import breeze.linalg.DenseVector
import core.NeuralNetwork
import data.DataSet

import scala.collection.SeqView

object Trainer {

    def train(net: NeuralNetwork, dataSet: DataSet, epochs: Int): Unit = {

        for (epoch <- 1 to epochs) {
            val trainingDataSet = dataSet.trainingInput zip dataSet.trainingOutput
            for (((inputVector, targetOutputs), index) <- trainingDataSet.view.zipWithIndex) {

                val actualOutputs = net.feedForward(inputVector)

                // Note: backPropagate is stateful
                net.backPropagate(actualOutputs, targetOutputs)
            }

            val score = evaluate(net, dataSet.testingInput, dataSet.testingOutput)
            println(s"${epoch}: ${score}/${dataSet.testingInput.size}")
        }

        println(net)
    }

    private def evaluate(net: NeuralNetwork, inputsList: Seq[DenseVector[Double]], outputList: Seq[DenseVector[Double]]): Int = {

        val inputsAndTargets: SeqView[((DenseVector[Double], DenseVector[Double]), Int), Seq[_]] = (inputsList zip outputList).view.zipWithIndex

        inputsAndTargets map {
            case((inputs, targetOutputs), index) => {
                val actualOutputs = net.feedForward(inputs)
                findIndexOfMaxElementIn(actualOutputs) == findIndexOfMaxElementIn(targetOutputs)
            }
        } count(b => b)
    }

    private def findIndexOfMaxElementIn(vector: DenseVector[Double]): Int = {
        vector.toArray.zipWithIndex.maxBy(_._1)._2
    }

}
