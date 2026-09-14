package scair.dialects.linalg.canonicalization

import scair.dialects.builtin.*
import scair.dialects.linalg.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.CanonicalizationPatterns

// ██╗░░░░░ ██╗ ███╗░░██╗ ░█████╗░ ██╗░░░░░ ░██████╗░
// ██║░░░░░ ██║ ████╗░██║ ██╔══██╗ ██║░░░░░ ██╔════╝░
// ██║░░░░░ ██║ ██╔██╗██║ ███████║ ██║░░░░░ ██║░░██╗░
// ██║░░░░░ ██║ ██║╚████║ ██╔══██║ ██║░░░░░ ██║░░╚██╗
// ███████╗ ██║ ██║░╚███║ ██║░░██║ ███████╗ ╚██████╔╝
// ╚══════╝ ╚═╝ ╚═╝░░╚══╝ ╚═╝░░╚═╝ ╚══════╝ ░╚═════╝░
//
// ░█████╗░ ░█████╗░ ███╗░░██╗ ░█████╗░ ███╗░░██╗ ██╗ ░█████╗░ ░█████╗░ ██╗░░░░░ ██╗ ███████╗ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔══██╗ ████╗░██║ ██║ ██╔══██╗ ██╔══██╗ ██║░░░░░ ██║ ╚════██║ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║
// ██║░░╚═╝ ███████║ ██╔██╗██║ ██║░░██║ ██╔██╗██║ ██║ ██║░░╚═╝ ███████║ ██║░░░░░ ██║ ░░███╔═╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║
// ██║░░██╗ ██╔══██║ ██║╚████║ ██║░░██║ ██║╚████║ ██║ ██║░░██╗ ██╔══██║ ██║░░░░░ ██║ ██╔══╝░░ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║
// ╚█████╔╝ ██║░░██║ ██║░╚███║ ╚█████╔╝ ██║░╚███║ ██║ ╚█████╔╝ ██║░░██║ ███████╗ ██║ ███████╗ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║
// ░╚════╝░ ╚═╝░░╚═╝ ╚═╝░░╚══╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═╝ ░╚════╝░ ╚═╝░░╚═╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝

/** Matches a variadic operand group whose values are all tensors.
  *
  * Stands in for MLIR's `hasPureTensorSemantics`, which ScaIR's method-free port
  * of the dialect does not provide.
  */
object Tensors:

  def unapply(values: Seq[Value[Attribute]]): Option[Seq[Value[Attribute]]] =
    Option.when(values.forall(_.typ.isInstanceOf[TensorType]))(values)

/** Remove `linalg.generic` operations that just copy the values from inputs to
  * results. In the memref case, the operation must be copying to and from the
  * same value. Requirements are:
  *   1. All iterator types are parallel
  *   1. The body contains just a yield operation with the yielded values being
  *      the arguments corresponding to the operands.
  *
  * MLIR: `EraseIdentityLinalgOp`.
  */
val EraseIdentityGeneric = pattern {

  // Buffer semantics: the op reads and writes the very same memref, so it is a
  // no-op and can simply go away.
  case Generic(
        inputs = Seq(input @ Value(_: MemrefType)),
        outputs = Seq(output),
        indexing_maps = ArrayAttribute(maps*),
        result_tensors = Seq(),
        region = Region(
          body @ Block(operations = BlockOperations(Yield(Seq(yielded))))
        ),
      )
      // All indexing maps must be equal. It follows that they are permutations.
      if maps.distinct.sizeIs <= 1
        // Expected single input and output to be the same value.
        && (input eq output)
        // Cannot fold a fill-like op, whose yielded value comes from elsewhere.
        && yielded.owner.contains(body) =>
    PatternAction.Erase

  // Tensor semantics: every result is one of the operands, forwarded through the
  // block argument the body yields.
  case op @ Generic(
        inputs = Tensors(_),
        outputs = Tensors(_),
        indexing_maps = ArrayAttribute(maps*),
        result_tensors = results,
        region = Region(
          body @ Block(operations = BlockOperations(Yield(yielded)))
        ),
      )
      // All indexing maps must be equal. It follows that they are permutations.
      if maps.distinct.sizeIs <= 1
        // Every yielded value must be an argument of the body, as its argument
        // number is the operand number to replace the uses of this op with.
        && yielded.forall(_.owner.contains(body))
        && yielded.sizeIs == results.size
        // TODO: the input can have a different type than the result, e.g. a
        // dynamic input dimension turned into a static output dimension. MLIR
        // bridges the two with a `tensor.cast`, or a `sparse_tensor.convert`
        // when either side carries a sparse encoding; ScaIR has neither dialect,
        // so the pattern bails out instead.
        && yielded.lazyZip(results).forall: (y, r) =>
          op.operands(body.arguments.indexOf(y)).typ == r.typ
      =>
    (Seq(), yielded.map(y => op.operands(body.arguments.indexOf(y))))
}

given CanonicalizationPatterns[Generic](
  EraseIdentityGeneric
)
