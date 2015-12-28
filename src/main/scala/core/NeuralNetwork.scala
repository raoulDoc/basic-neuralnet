package core

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.language.postfixOps


class NeuralNetwork(val inputLayer: InputLayer,
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

        val errorArray = (actualOutputs.toArray zip targetOutputs.toArray).map {
            case(a, t) => a * (1 - a) * (t - a)
        }

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

    private def updateWeights(fromLayer: Layer, toLayer: MutableLayer, deltas: DenseVector[Double]) : Unit = {
        val rows = fromLayer.getNumberOfUnits
        val cols = toLayer.getNumberOfUnits

        val newWeightsArray =
            (0 until cols).flatMap(j =>
                (0 until rows).map(i => {
                    val weight = toLayer.getWeightAtIndex(i, j)
                    val deltaWeight = learningRate * deltas (j) * fromLayer.getOutputAtIndex(i)
                    weight + deltaWeight
                }
                )
            ).toArray

        val newWeightsMatrix = new DenseMatrix(rows, cols, newWeightsArray)
        toLayer.updateWeights(newWeightsMatrix)
    }

    private def updateOutputsWeights(deltaOutputsErrors: DenseVector[Double]): Unit = {
        updateWeights(fromLayer = hiddenLayer, toLayer = outputLayer, deltaOutputsErrors)
    }

    private def updateOutputBiasFor(layer: MutableLayer, deltaErrors: DenseVector[Double]): Unit = {
        val newBiases = (0 until layer.getNumberOfUnits)
            .map(j => gammaBias * deltaErrors(j))
            .toArray

        layer.updateBiases(DenseVector[Double](newBiases))
    }

    private def updateOutputBias(deltaErrors: DenseVector[Double]): Unit = {
        updateOutputBiasFor(outputLayer, deltaErrors)
    }

    private def updateHiddenWeights(deltaHiddenErrors: DenseVector[Double]): Unit = {
        updateWeights(fromLayer = inputLayer, toLayer = hiddenLayer, deltaHiddenErrors)
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
        val inputLayer = new InputLayer(input)
        val hiddenLayer = new HiddenLayer(hidden, inputLayer)
        val outpoutLayer = new OutputLayer(output, hiddenLayer)

        new NeuralNetwork(inputLayer, hiddenLayer, outpoutLayer, learningRate, gammaBias)
    }
}
