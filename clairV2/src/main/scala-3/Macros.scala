package scair.clairV2.macros

import scala.quoted.*

inline def typeToString[T]: String = ${ typeToStringImpl[T] }

def typeToStringImpl[T: Type](using Quotes): Expr[String] = {
  import quotes.reflect.*
  Expr(Type.show[T])
}

inline def getClassPath[T]: String = ${ getClassPathImpl[T] }

def getClassPathImpl[T: Type](using Quotes): Expr[String] = {
  import quotes.reflect.*
  val classSymbol = TypeRepr.of[T].typeSymbol
  val classPath = classSymbol.fullName
  Expr(classPath)
}
