package data

import breeze.linalg.DenseVector

class DataSet(val trainingInput: Seq[DenseVector[Double]],
              val trainingOutput: Seq[DenseVector[Double]],
              val testingInput: Seq[DenseVector[Double]],
              val testingOutput: Seq[DenseVector[Double]]) {
}

object DataSet {

    def apply(trainingInput: Seq[DenseVector[Double]],
              trainingOutput: Seq[DenseVector[Double]],
              testingInput: Seq[DenseVector[Double]],
              testingOutput: Seq[DenseVector[Double]]) : DataSet = {

        new DataSet(trainingInput, trainingOutput, testingInput, testingOutput)
    }
}
