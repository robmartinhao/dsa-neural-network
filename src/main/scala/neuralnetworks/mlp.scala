package neuralnetworks

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gaussian

object mlp {

  // Inicializa os pesos da rede neural
  def initializeWeights(layerDimesions: Seq[Int]): Seq[DenseMatrix[Double]] = {

    layerDimesions.dropRight(1).zipWithIndex.foldLeft(new Array[DenseMatrix[Double]](layerDimesions.size - 1)) {
      case (allWeigths, (layerDim, layerIdx)) =>
        val layerDimWithBias = layerDim + 1

        val layerWeights = new DenseMatrix(layerDimWithBias, layerDimesions(layerIdx + 1),
          new Gaussian(0, 0.2).sample(layerDimWithBias * layerDimesions(layerIdx)).toArray
        )

        allWeigths(layerIdx) = layerWeights

        allWeigths
    }
  }
}
