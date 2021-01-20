package exemplos

import breeze.linalg.DenseMatrix

object AlgebraLinearExemplo extends App {

  // Cria uma matriz densa
  val dm1 = DenseMatrix((1.0, 2.0), (3.0, 4.0))

  // Matriz Transposta
  val dm1Transpose = dm1.t

  println(s"${dm1} transposed is ${dm1Transpose}")

  // Cria outra matriz densa
  val dm2 = DenseMatrix((5.0, 6.0), (7.0, 8.0))

  // Produto das
  val matrixProduct = dm1 * dm2

  println(s"The product of ${dm1} and ${dm2} is ${matrixProduct}")

  // Soma elementwise
  val matrixElSum = dm1 + dm2

  println(s"A soma elementwise de ${dm1} e ${dm2} = ${matrixElSum}")
}
