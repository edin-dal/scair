package scair.transformations.patterns

import scair.ir.*
import scair.transformations.*

//
// ██████╗░ ░█████╗░ ████████╗ ████████╗ ███████╗ ██████╗░ ███╗░░██╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ████╗░██║ ██╔════╝
// ██████╔╝ ███████║ ░░░██║░░░ ░░░██║░░░ █████╗░░ ██████╔╝ ██╔██╗██║ ╚█████╗░
// ██╔═══╝░ ██╔══██║ ░░░██║░░░ ░░░██║░░░ ██╔══╝░░ ██╔══██╗ ██║╚████║ ░╚═══██╗
// ██║░░░░░ ██║░░██║ ░░░██║░░░ ░░░██║░░░ ███████╗ ██║░░██║ ██║░╚███║ ██████╔╝
// ╚═╝░░░░░ ╚═╝░░╚═╝ ░░░╚═╝░░░ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚══╝ ╚═════╝░
//

enum PatternAction:
  case Erase
  case Abort

/** Defines a RewritePattern from a partial function. The partial function can
  * return the following types:
  *   - `Unit`: to erase the operation
  *   - `Operation`: to replace the operation with a single new operation
  *   - `Seq[Operation]`: to replace the operation with multiple new operations
  *   - `(Operation | Seq[Operation], Value[?] | Seq[Value[?]])`: to replace the
  *     operation with new operations and new results
  *
  * @return
  *   A RewritePattern to hook to the infrastructure.
  */
inline def pattern(
    partial: PartialFunction[
      Operation,
      PatternAction | Operation | Seq[Operation] |
        (Operation | Seq[Operation], Value[?] | Seq[Value[?]])
    ]
): RewritePattern =
  val lifted = partial.lift

  object pattern extends RewritePattern:
    override def match_and_rewrite(
        op: Operation,
        rewriter: PatternRewriter
    ): Unit =
      lifted(op).map({ (output) =>
        output match
          case PatternAction.Erase =>
            rewriter.erase_op(op)
          case PatternAction.Abort => ()
          case both: (Operation | Seq[Operation], Value[?] | Seq[Value[?]]) =>
            rewriter.replace_op(
              op,
              both._1,
              Some(both._2 match
                case r: Value[?]       => Seq(r)
                case rs: Seq[Value[?]] => rs)
            )
          case new_op: Operation =>
            rewriter.replace_op(op, new_op, None)
          case new_ops: Seq[Operation @unchecked] =>
            rewriter.replace_op(op, new_ops, None)
      })

  pattern

object Owner:
  def unapply(v: Value[Attribute]) = v.owner
