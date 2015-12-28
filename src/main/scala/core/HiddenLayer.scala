package core

import breeze.linalg.{DenseMatrix, DenseVector}

class HiddenLayer(val size: Int,
                  val previous: InputLayer,
                  var weights: DenseMatrix[Double],
                  var biases: DenseVector[Double]) extends MutableLayer {

    assert(size == weights.cols, "Check number of units matches weights matrix")
    var outputs = DenseVector.zeros[Double](size)

    def this(size: Int, previous: InputLayer) = this(size,
                                                     previous,
                                                     DenseMatrix.rand(previous.size, size),
                                                     DenseVector.rand(size))

    def calculateOutputs(input: DenseVector[Double]) : DenseVector[Double] = {
        val array = (0 until size).map(getOutputsForNeuronAtIndex(_, input)).toArray
        val newOutputs = DenseVector[Double](array)
        outputs = newOutputs
        outputs
    }

    private def getOutputsForNeuronAtIndex(index: Int, inputs: DenseVector[Double]): Double = {

        val weightsForNeuron = extractWeightsForNeuronAtIndex(index)
        val biasForNeuron = biases(index)
        Neuron.activate(inputs, weightsForNeuron, biasForNeuron)
    }

    private def extractWeightsForNeuronAtIndex(index: Int): DenseVector[Double] = {
        // extract column values
        weights(::, index)
    }

    override def toString() : String = {
        "Weights:\n" +
        weights.toString + "\n\n" +
        "Biases:\n" +
        biases.toString
    }

    override def getNumberOfUnits(): Int = size

    override def updateWeights(newWeights: DenseMatrix[Double]): Unit = {
        weights = newWeights
    }

    override def updateBiases(newBiases: DenseVector[Double]): Unit = {
        biases = newBiases
    }

    override def getOutputAtIndex(index: Int): Double = outputs(index)

    override def getWeightAtIndex(i: Int, j: Int): Double = weights(i, j)
}
