package data

import java.net.URL

import breeze.linalg.DenseVector

import scala.util.Random

object IrisDataSetLoader extends DataSetLoader {

    def load(trainingSplit: Double) : DataSet = {
        val dataSetUrl : URL = getClass.getResource("/iris-normalised")
        val lines = scala.io.Source.fromFile(dataSetUrl.toURI).getLines

        val endOfFeaturesIndex = 4
        val endOfOutputsIndex = 7

        val data : Seq[(DenseVector[Double], DenseVector[Double])] =
            lines.map(line => {
                val details = line.split("\\s+").map(_.toDouble)

                val input = DenseVector[Double](details.slice(0, endOfFeaturesIndex))
                val output = DenseVector[Double](details.slice(endOfFeaturesIndex, endOfOutputsIndex))

                (input, output)

            }).toList

        splitData(trainingSplit, data)
    }

    private def splitData(trainingSplit: Double, data: Seq[(DenseVector[Double], DenseVector[Double])]): DataSet = {
        val mixedData = Random.shuffle(data)

        val splitIndex: Int = (data.size * trainingSplit).toInt

        val (trainingInput, trainingOutput) = mixedData.slice(0, splitIndex).unzip
        val (testingInput, testingOutput) = mixedData.slice(splitIndex, data.size).unzip

        DataSet(trainingInput, trainingOutput, testingInput, testingOutput)
    }
}
