package core

import breeze.linalg.{DenseMatrix, DenseVector}

trait Layer {
    def getNumberOfUnits(): Int
    def updateWeights(newWeights: DenseMatrix[Double])
    def updateBiases(newBiases: DenseVector[Double]): Unit
}
