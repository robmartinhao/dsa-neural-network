package neuralnetworks

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gaussian
import spire.implicits.rightModuleOps

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

  def backwardPass(layerActivations: Array[DenseMatrix[Double]],
                   layerWeights: Seq[DenseMatrix[Double]],
                   networkOutput: DenseMatrix[Double],
                   target: DenseMatrix[Double],
                   activationFn: DenseMatrix[Double] => DenseMatrix[Double],
                   activationFnDerivative: DenseMatrix[Double] => DenseMatrix[Double]): Array[DenseMatrix[Double]] =
  {

    val predictionError = target - networkOutput

    val innerActivations = layerActivations.dropRight(1)

    innerActivations.zip(layerWeights).zipWithIndex.foldRight(
      (new Array[DenseMatrix[Double]](layerActivations.size - 1), predictionError))
    { case (((activation, weight), idx), (gradients, delta)) =>

      val nextDelta = activationFnDerivative(activation) :* (delta * weight.t(::, 1 to -1))

      val activationWithBias = DenseMatrix.horzcat(
        DenseMatrix(1.0),
        activationFn(activation)
      )

      val layerGradient = delta.t * activationWithBias

      gradients(idx) = layerGradient

      (gradients, nextDelta)

    }._1

  }

}
