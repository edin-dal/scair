package scair.interpreter

object printf extends ExtFuncImpl:
  def name: String = "printf"

  def run(args: Seq[Any]): Unit = 0