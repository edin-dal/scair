package scair.macros

import scala.quoted.*


def varOfChainImpl(a: Expr[(Quotes) => Expr[Unit] => Expr[Unit]], b: Expr[(Quotes) => Expr[Unit] => Expr[Unit]], s: Expr[Expr[Unit]])(using Quotes) : Expr[(Quotes) => Expr[Unit]] =
  // val unit = Expr(())
  '{
    (q: Quotes) =>
      val bb = ${b}(q)
      val aa = ${a}(q)
      aa(bb(${s}))
  }