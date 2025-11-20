package scair.interpreter

case class ShapedArray(
    private val data: Array[Any],
    shape: Seq[Int]
):

  lazy val strides: Seq[Int] =
    shape.scanRight(1)(_ * _).tail

  def length: Int =
    shape.product

  private def offset(indices: Seq[Int]): Int =
    indices.zip(strides).map(_ * _).sum

  def apply(indices: Seq[Int]): Any = data(offset(indices))

  def update(indices: Seq[Int], value: Any): Unit =
    data(offset(indices)) = value
