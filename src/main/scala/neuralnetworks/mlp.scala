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

  def forwardPass(networkInput: DenseMatrix[Double],
                  layerWeights: Seq[DenseMatrix[Double]],
                  activationFn: DenseMatrix[Double] => DenseMatrix[Double]): (Array[DenseMatrix[Double]], DenseMatrix[Double]) = {
    assert(networkInput.rows + 1 == layerWeights.head.rows)

    val initialActivations = new Array[DenseMatrix[Double]](layerWeights.size + 1)

    initialActivations(0) = networkInput.t

    layerWeights.zipWithIndex.foldLeft((initialActivations, networkInput.t)) {
      case ((activations, input), (weight, weightIdx)) =>

        val inputWithBias = DenseMatrix.horzcat(DenseMatrix(1.0), input)

        val layerActivation = inputWithBias * weight

        activations(weightIdx + 1) = layerActivation

        (activations, activationFn(layerActivation))
    }
  }
}
