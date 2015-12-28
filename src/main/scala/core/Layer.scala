package core

/**
  * Created by raoul-gabrielurma on 28/12/2015.
  */
trait Layer {
    def getNumberOfUnits(): Int
    def getOutputAtIndex(index: Int) : Double
}
