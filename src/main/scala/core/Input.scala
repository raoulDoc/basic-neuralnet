package core

import breeze.linalg.DenseVector

class Input(val size: Int) {
    var outputs = DenseVector.zeros[Double](size)
}
