package scair.core.macros

import scair.ir.DataAttribute

import scala.quoted.*

//
// ████████╗ ██████╗░ ░█████╗░ ███╗░░██╗ ░██████╗ ██████╗░ ░█████╗░ ██████╗░ ███████╗ ███╗░░██╗ ████████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ████╗░██║ ╚══██╔══╝
// ░░░██║░░░ ██████╔╝ ███████║ ██╔██╗██║ ╚█████╗░ ██████╔╝ ███████║ ██████╔╝ █████╗░░ ██╔██╗██║ ░░░██║░░░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║╚████║ ░╚═══██╗ ██╔═══╝░ ██╔══██║ ██╔══██╗ ██╔══╝░░ ██║╚████║ ░░░██║░░░
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║░╚███║ ██████╔╝ ██║░░░░░ ██║░░██║ ██║░░██║ ███████╗ ██║░╚███║ ░░░██║░░░
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝░░╚══╝ ╚═════╝░ ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚══════╝ ╚═╝░░╚══╝ ░░░╚═╝░░░
//
// ██████╗░ ░█████╗░ ████████╗ ░█████╗░
// ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██╔══██╗
// ██║░░██║ ███████║ ░░░██║░░░ ███████║
// ██║░░██║ ██╔══██║ ░░░██║░░░ ██╔══██║
// ██████╔╝ ██║░░██║ ░░░██║░░░ ██║░░██║
// ╚═════╝░ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝░░╚═╝
//

/** Type helper to extract the type of the data from the DataAttribute.
  */
type DataTypeOf[T] = T match
  case DataAttribute[t] => t

/** Typeclass to provide automatic construction of a DataAttribute from its
  * datatype.
  */
trait TransparentData[T <: DataAttribute[?]]
    extends Conversion[DataTypeOf[T], T]:
  def apply(data: DataTypeOf[T]): T = attrConversion(data)
  def attrConversion(data: DataTypeOf[T]): T

object TransparentData:
  inline def derived[T <: DataAttribute[?]] = ${ derivedImpl[T] }

  def derivedImpl[T <: DataAttribute[?]: Type](using Quotes) =
    import quotes.reflect.*

    Type.of[T] match
      case '[DataAttribute[t]] =>
        '{
          new TransparentData[T]:

            def attrConversion(data: t): T = ${
              // Return a call to the primary constructor of the ADT.
              Apply(
                Select(
                  New(TypeTree.of[T]),
                  TypeRepr.of[T].typeSymbol.primaryConstructor,
                ),
                List('{ data }.asTerm),
              ).asExprOf[T]
            }
        }
