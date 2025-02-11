package test.showmacros

// import scair.ir.{Value, Attribute, ListType, DictType}

// import test.mirrored.*
// import scala.reflect.*
// import scala.compiletime.*
// import scala.deriving.*
import scala.quoted.*

inline def typeToString[T]: String = ${ typeToStringImpl[T] }

def typeToStringImpl[T: Type](using Quotes): Expr[String] = {
  import quotes.reflect.*
  Expr(Type.show[T])
}

// import scala.quoted.*

// object MLIRRealm {

//   inline given derived[T <: ADTOperation](using
//       m: Mirror.ProductOf[T]
//   ): MLIRRealm[T] = ${ deriveMLIRRealm[T] }

//   def deriveMLIRRealm[T <: ADTOperation: Type](using
//       Quotes
//   ): Expr[MLIRRealm[T]] = {
//     import quotes.reflect.*

//     val tpe = TypeRepr.of[T]
//     val className = tpe.typeSymbol.name

//     val fields = tpe.typeSymbol.caseFields.map { field =>
//       val name = field.name
//       val fieldType = field.tree match {
//         case v: ValDef => v.tpt.tpe
//         case _         => report.throwError(s"Unexpected field type for $name")
//       }
//       (name, fieldType)
//     }

//     /*≡≡=---=≡≡≡≡≡≡≡≡=---=≡≡*\
//     ||   UNVERIFY METHODS   ||
//     \*≡==----=≡≡≡≡≡≡=----==≡*/

//     val unverifyOperands = fields.collect {
//       case (name, fieldType) if fieldType <:< TypeRepr.of[Operand[?]] => name
//     }

//     val unverifyBody = fields.map { case (name, fieldType) =>
//       if (fieldType <:< TypeRepr.of[Operand[?]]) {
//         '{
//           op.asInstanceOf[T]
//             .getClass
//             .getMethod(${ Expr(name) })
//             .invoke(op)
//             .asInstanceOf[Value[Attribute]]
//         }
//       } else if (fieldType <:< TypeRepr.of[Result[?]]) {
//         '{
//           op.asInstanceOf[T]
//             .getClass
//             .getMethod(${ Expr(name) })
//             .invoke(op)
//             .asInstanceOf[Value[Attribute]]
//         }
//       } else {
//         report.throwError(s"Unsupported field type for $name")
//       }
//     }

//     /*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
//     ||   VERIFY METHODS   ||
//     \*≡==----=≡≡≡≡=----==≡*/

//     val verifyBody = fields.zipWithIndex.map { case ((name, fieldType), idx) =>
//       if (fieldType <:< TypeRepr.of[Operand[?]]) {
//         '{
//           op.operands(${ Expr(idx) })
//             .asInstanceOf[Value[${ fieldType.typeArgs.head.show }]]
//         }
//       } else if (fieldType <:< TypeRepr.of[Result[?]]) {
//         '{
//           op.results(${ Expr(idx) })
//             .asInstanceOf[Value[${ fieldType.typeArgs.head.show }]]
//         }
//       } else {
//         report.throwError(s"Unsupported field type for $name")
//       }
//     }

//     '{
//       new MLIRRealm[T] {
//         def unverify(op: T): RegisteredOp[T] = {
//           val op1 = RegisteredOp[T](
//             name = ${ Expr(className) },
//             operands = ListType(
//               $unverifyBody
//             )
//           )

//           op1.results.clear()
//           op1.results.addAll(ListType(op.result))
//           op1
//         }

//         def verify(op: RegisteredOp[_]): T = {
//           ${ Expr(className) }(
//             $verifyBody
//           )
//         }
//       }
//     }
//   }

// }
