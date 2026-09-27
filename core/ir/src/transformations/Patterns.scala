package scair.transformations

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

/** What a pattern returns: the operations replacing the matched one, with their
  * replacing results given explicitly or taken from the last of them, or a
  * [[PatternAction]].
  */
type RewriteResult = PatternAction | Operation | Seq[Operation] |
  (Operation | Seq[Operation], Value[?] | Seq[Value[?]])

/** One operation or several, as several. */
def asOps(ops: Operation | Seq[Operation]): Seq[Operation] =
  ops match
    case op: Operation => Seq(op)
    case ops: Seq[?]   => ops.asInstanceOf[Seq[Operation]]

/** The operations a rewrite result replaces with, and the results to replace
  * with - `None` where the result leaves them to be taken from the last
  * operation. Erasure and abortion are not decoded here; match them first.
  */
def asReplacement(
    result: RewriteResult
): (Seq[Operation], Option[Seq[Value[Attribute]]]) =
  result match
    case (ops, results): (Operation | Seq[Operation], ?) =>
      (
        asOps(ops),
        Some(results match
          case r: Value[?]       => Seq(r.asInstanceOf[Value[Attribute]])
          case rs: Seq[Value[?]] => rs.asInstanceOf[Seq[Value[Attribute]]]),
      )
    case ops: (Operation | Seq[Operation]) => (asOps(ops), None)
    case action: PatternAction             =>
      throw new Exception(s"$action is not a replacement.")

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
    inline partial: PartialFunction[Operation, RewriteResult]
): RewritePattern =
  object pattern extends RewritePattern:
    override def matchAndRewrite(
        op: Operation,
        rewriter: PatternRewriter,
    ): Unit =
      partial.applyOrElse(op, (_: Operation) => PatternAction.Abort) match
        case PatternAction.Erase => rewriter.eraseOp(op)
        case PatternAction.Abort => ()
        case result              =>
          val (newOps, newResults) = asReplacement(result)
          rewriter.replaceOp(op, newOps, newResults)

  pattern

object Owner:
  def unapply(v: Value[Attribute]) = v.owner
