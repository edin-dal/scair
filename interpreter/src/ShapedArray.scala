package scair.interpreter

case class ShapedArray(
    shape: Seq[Int]
):

  private val data: Array[Any] = Array
    .fill(shape.product)(0) // default value of 0 for all elements

  require(shape.forall(_ >= 0), "Shape dimensions must be non-negative")

  require(
    shape.product == data.length,
    s"Data length ${data.length} doesn't match shape $shape",
  )

  lazy val strides: Seq[Int] =
    shape.scanRight(1)(_ * _).tail

  def length: Int =
    shape.product

  private def offset(indices: Seq[Int]): Int =
    indices.zip(strides).map(_ * _).sum

  def apply(indices: Seq[Int]): Any = data(offset(indices))

  def update(indices: Seq[Int], value: Any): Unit =
    data(offset(indices)) = value
