package core

import breeze.linalg.{DenseMatrix, DenseVector}

trait MutableLayer extends Layer {
    def updateWeights(newWeights: DenseMatrix[Double])
    def updateBiases(newBiases: DenseVector[Double]): Unit
    def getWeightAtIndex(i: Int, j: Int) : Double
}
