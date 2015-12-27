package core

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.language.postfixOps


class NeuralNetwork(val inputLayer: Input,
                    val hiddenLayer: HiddenLayer,
                    val outputLayer: OutputLayer,
                    val learningRate: Double,
                    val gammaBias: Double) {

    def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
        inputLayer.outputs = input
        val outputsFromHiddenLayer = hiddenLayer.calculateOutputs(input)
        outputLayer.calculateOutputs(outputsFromHiddenLayer)
    }

    def backPropagate(actualOutputs: DenseVector[Double], targetOutputs: DenseVector[Double]): Unit = {
        val outputsErrors = calculateOutputsErrors(actualOutputs, targetOutputs)
        val hiddenErrors = calculateHiddenErrors(outputsErrors)

        updateOutputsWeights(outputsErrors)
        updateOutputBias(outputsErrors)

        updateHiddenWeights(hiddenErrors)
        updateHiddenBias(hiddenErrors)
    }

    def calculateOutputsErrors(actualOutputs: DenseVector[Double],
                               targetOutputs: DenseVector[Double]): DenseVector[Double] = {

        // TODO: would love a zip
        val errorArray = (0 until outputLayer.size).map( k => {
            val a = actualOutputs(k)
            val t = targetOutputs(k)
            a * (1 - a) * (t - a)
        }).toArray

        DenseVector[Double](errorArray)
    }

    def calculateHiddenErrors(deltaOutputsErrors: DenseVector[Double]): DenseVector[Double] = {
        val hiddenErrorArray = (0 until hiddenLayer.size).map( h => {
            val d = hiddenLayer.outputs(h) * (1 - hiddenLayer.outputs(h))
            // returns a row but we want a column hence the .transpose
            val weightsConnectedToHiddenUnit: DenseVector[Double] = outputLayer.weights(h, ::).t
            val product = weightsConnectedToHiddenUnit dot deltaOutputsErrors
            d * product
        }).toArray

        DenseVector[Double](hiddenErrorArray)
    }

    private def updateOutputsWeights(deltaOutputsErrors: DenseVector[Double]): Unit = {
        //TODO: better way of doing this
        val newWeights = DenseMatrix.zeros[Double](hiddenLayer.size, outputLayer.size)

        (0 until outputLayer.size) map { j =>
            (0 until hiddenLayer.size) map { i => {
                val weight = outputLayer.weights(i, j)
                val deltaWeight = learningRate * deltaOutputsErrors(j) * hiddenLayer.outputs(i)
                newWeights(i, j) = weight + deltaWeight
            }
            }
        }
        outputLayer.updateWeights(newWeights)
    }

    private def updateOutputBiasFor(layer: Layer, deltaErrors: DenseVector[Double]): Unit = {
        val newBiases = (0 until layer.getNumberOfUnits)
            .map(j => gammaBias * deltaErrors(j))
            .toArray

        layer.updateBiases(DenseVector[Double](newBiases))
    }

    private def updateOutputBias(deltaErrors: DenseVector[Double]): Unit = {
        updateOutputBiasFor(outputLayer, deltaErrors)
    }

    private def updateHiddenWeights(deltaHiddenErrors: DenseVector[Double]): Unit = {
        //TODO: better way of doing this
        val newWeights = DenseMatrix.zeros[Double](inputLayer.size, hiddenLayer.size)

        (0 until hiddenLayer.size) map { j =>
            (0 until inputLayer.size) map { i => {
                val weight = hiddenLayer.weights(i, j)
                val deltaWeight = learningRate * deltaHiddenErrors(j) * inputLayer.outputs(i)
                newWeights(i, j) = weight + deltaWeight
            }
            }
        }

        hiddenLayer.updateWeights(newWeights)
    }

    private def updateHiddenBias(deltaErrors: DenseVector[Double]): Unit = {
        updateOutputBiasFor(hiddenLayer, deltaErrors)
    }

    override def toString() : String = {
        s"Neural network of ${inputLayer.size}x" +
        s"${hiddenLayer.getNumberOfUnits()}x" +
        s"${outputLayer.getNumberOfUnits()}\n" +
        "------\n" +
        "Hidden core.Layer\n" +
        s"${hiddenLayer}\n" +
        "------\n" +
        "Output core.Layer\n" +
        s"${outputLayer}\n" +
        "------\n"
    }
}

object NeuralNetwork {

    def apply(input: Int,
              hidden: Int,
              output: Int,
              learningRate: Double = 3.0,
              gammaBias: Double = 1.0): NeuralNetwork = {
        val inputLayer = new Input(input)
        val hiddenLayer = new HiddenLayer(hidden, inputLayer)
        val outpoutLayer = new OutputLayer(output, hiddenLayer)

        new NeuralNetwork(inputLayer, hiddenLayer, outpoutLayer, learningRate, gammaBias)
    }
}
