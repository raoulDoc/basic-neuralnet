package data

trait DataSetLoader {
    def load(trainingSplit: Double = 0.5) : DataSet
}
