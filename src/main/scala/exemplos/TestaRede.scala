package exemplos

import neuralnetworks.{mlp, TrainingExample}
import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid

object TestaRede extends App {

  val dataset = Seq(
    TrainingExample(DenseMatrix(0.0, 0.0), DenseMatrix(0.0)),
    TrainingExample(DenseMatrix(1.0, 0.0), DenseMatrix(1.0)),
    TrainingExample(DenseMatrix(0.0, 1.0), DenseMatrix(1.0)),
    TrainingExample(DenseMatrix(1.0, 1.0), DenseMatrix(0.0))
  )

  val actFn: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)
  val actFnDerivate = (x: DenseMatrix[Double]) => x.map(y => sigmoid(y) * (1 - sigmoid(y)))

  val learningRate = 0.1;

  mlp.train(dataset,
    Seq(2, 2, 1),
    actFn,
    actFnDerivate,
    learningRate,
    50000)
}
