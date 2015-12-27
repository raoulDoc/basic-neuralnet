package core

import breeze.linalg.DenseVector
import breeze.numerics._

/**
  * Created by raoul-gabrielurma on 27/12/2015.
  */
object Neuron {

    // return output of one neuron
    // TODO: extract activation function and derivatives
    def activate(inputs: DenseVector[Double],
                 weights: DenseVector[Double], bias: Double) : Double = {

        val value: Double = inputs dot weights
        sigmoid(value + bias)
    }
}
