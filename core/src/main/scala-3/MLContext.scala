package scair

import scair.ir.*

import scala.collection.mutable

// ███╗░░░███╗ ██╗░░░░░ ░█████╗░ ░█████╗░ ███╗░░██╗ ████████╗ ███████╗ ██╗░░██╗ ████████╗
// ████╗░████║ ██║░░░░░ ██╔══██╗ ██╔══██╗ ████╗░██║ ╚══██╔══╝ ██╔════╝ ╚██╗██╔╝ ╚══██╔══╝
// ██╔████╔██║ ██║░░░░░ ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ░╚███╔╝░ ░░░██║░░░
// ██║╚██╔╝██║ ██║░░░░░ ██║░░██╗ ██║░░██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ░██╔██╗░ ░░░██║░░░
// ██║░╚═╝░██║ ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ░░░██║░░░ ███████╗ ██╔╝╚██╗ ░░░██║░░░
// ╚═╝░░░░░╚═╝ ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ░░░╚═╝░░░

class MLContext() {

  val dialectOpContext: mutable.Map[String, OperationObject] = mutable.Map()
  val dialectAttrContext: mutable.Map[String, AttributeObject] = mutable.Map()

  def getOperation(name: String) = dialectOpContext.get(name)

  def getAttribute(name: String) = dialectAttrContext.get(name)

  def registerDialect(dialect: Dialect) = {
    dialectOpContext ++= {
      for { dialectOp <- dialect.operations } yield {
        dialectOp.name -> dialectOp
      }
    }

    dialectAttrContext ++= {
      for { dialectAttr <- dialect.attributes } yield {
        dialectAttr.name -> dialectAttr
      }
    }
  }
}
