package scair

import scair.ir.*
import scair.transformations.ModulePass

import scala.collection.mutable

// ███╗░░░███╗ ██╗░░░░░ ░█████╗░ ░█████╗░ ███╗░░██╗ ████████╗ ███████╗ ██╗░░██╗ ████████╗
// ████╗░████║ ██║░░░░░ ██╔══██╗ ██╔══██╗ ████╗░██║ ╚══██╔══╝ ██╔════╝ ╚██╗██╔╝ ╚══██╔══╝
// ██╔████╔██║ ██║░░░░░ ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ░╚███╔╝░ ░░░██║░░░
// ██║╚██╔╝██║ ██║░░░░░ ██║░░██╗ ██║░░██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ░██╔██╗░ ░░░██║░░░
// ██║░╚═╝░██║ ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ░░░██║░░░ ███████╗ ██╔╝╚██╗ ░░░██║░░░
// ╚═╝░░░░░╚═╝ ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ░░░╚═╝░░░

class MLContext():

  val passContext: mutable.Map[String, ModulePass] = mutable.Map()

  def getPass(name: String) = passContext.get(name)

  def registerPass(passFactory: MLContext => ModulePass) =
    val pass = passFactory(this)
    passContext += pass.name -> pass

  val dialectOpContext: mutable.Map[String, OperationCompanion[?]] =
    mutable.Map()

  val dialectAttrContext: mutable.Map[String, AttributeCompanion[?]] =
    mutable.Map()

  def getOpCompanion(
      name: String,
      allowUnregisteredDialect: Boolean = false
  ) = dialectOpContext.get(name) match
    case Some(companion) => Right(companion)
    case None            =>
      if allowUnregisteredDialect then Right(UnregisteredOperation(name))
      else
        Left(
          s"Operation ${name} is not registered. If this is intended, use `--allow-unregistered-dialect`."
        )

  def getAttrCompanion(name: String) = dialectAttrContext.get(name)

  def registerDialect(dialect: Dialect) =
    dialectOpContext ++= {
      for dialectOp <- dialect.operations yield dialectOp.name -> dialectOp
    }

    dialectAttrContext ++= {
      for dialectAttr <- dialect.attributes
      yield dialectAttr.name -> dialectAttr
    }
