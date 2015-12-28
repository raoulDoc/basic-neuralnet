package core

import breeze.linalg.DenseVector

class InputLayer(val size: Int) extends Layer {
    var outputs = DenseVector.zeros[Double](size)

    override def getNumberOfUnits(): Int = size
    override def getOutputAtIndex(index: Int): Double = outputs(index)
}
