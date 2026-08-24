package scair.dialects.linalg

import scair.ir.*

// ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░ ███████╗ ░█████╗░ ░█████╗░ ███████╗ ░██████╗
// ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝
// ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝ █████╗░░ ███████║ ██║░░╚═╝ █████╗░░ ╚█████╗░
// ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗ ██╔══╝░░ ██╔══██║ ██║░░██╗ ██╔══╝░░ ░╚═══██╗
// ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║ ██║░░░░░ ██║░░██║ ╚█████╔╝ ███████╗ ██████╔╝
// ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░╚═╝ ░╚════╝░ ╚══════╝ ╚═════╝░

/* Op interfaces referenced by the Linalg operation definitions. Upstream each of
 * these carries a body of interface methods; here they are plain marker traits,
 * so that the ported operations can mirror MLIR's trait lists exactly. The
 * behaviour behind them is left for a later point in time.
 *
 * Interfaces owned by the Linalg dialect itself (LinalgInterfaces.td and
 * RelayoutOpInterface.td) come first, followed by the dialect-independent ones
 * that Linalg ops attach to. */

/*≡==--==≡≡≡≡≡==--=≡≡*\
||   LINALG-OWNED    ||
\*≡==---==≡≡≡==---==≡*/

/** MLIR: `LinalgStructuredInterface`. Common interface of every structured op,
  * giving access to its indexing maps, iterator types and operand roles.
  */
trait LinalgStructuredInterface extends Operation

/** MLIR: `LinalgContractionOpInterface`. A structured op whose body is a
  * multiply-accumulate over a permutation of its indexing maps.
  */
trait LinalgContractionOpInterface extends Operation

/** MLIR: `LinalgConvolutionOpInterface`. A structured op whose body is a
  * convolution.
  */
trait LinalgConvolutionOpInterface extends Operation

/** MLIR: `LinalgFillOpInterface`. A structured op that fills its output with a
  * single scalar value.
  */
trait LinalgFillOpInterface extends Operation

/** MLIR: `LinalgRelayoutOpInterface` (`RelayoutOpInterface.td`). An op that
  * changes the data layout of a tensor without changing its contents.
  */
trait LinalgRelayoutOpInterface extends Operation

/** MLIR: `AggregatedOpInterface`. An op that can be decomposed into a sequence
  * of simpler ops.
  */
trait AggregatedOpInterface extends Operation

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  DIALECT-INDEPENDENT  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

/** MLIR: `DestinationStyleOpInterface`. An op whose trailing operands are the
  * destinations its results are written into.
  */
trait DestinationStyleOpInterface extends Operation

/** MLIR: `ReifyRankedShapedTypeOpInterface`. An op that can materialise the
  * shapes of its ranked shaped results.
  */
trait ReifyRankedShapedTypeOpInterface extends Operation

/** MLIR: `TilingInterface`. An op that can be tiled into a loop nest over
  * smaller instances of itself.
  */
trait TilingInterface extends Operation

/** MLIR: `ConditionallySpeculatable`. An op whose speculatability depends on
  * its operands rather than being fixed.
  */
trait ConditionallySpeculatable extends Operation

/** MLIR: `MemoryEffectsOpInterface`. An op that declares its memory effects.
  */
trait MemoryEffectsOpInterface extends Operation

/** MLIR: `RecursiveMemoryEffects`. The op's memory effects are those of the ops
  * nested inside its regions.
  */
trait RecursiveMemoryEffects extends Operation

/** MLIR: `OpAsmOpInterface`. An op that customises how the assembly printer
  * names its results and block arguments.
  */
trait OpAsmOpInterface extends Operation

/** MLIR: `ReturnLike`. An op that returns from the region enclosing it.
  */
trait ReturnLike extends Operation

/** MLIR: `SingleBlockImplicitTerminator<"YieldOp">`. The op's region holds a
  * single block, terminated by `linalg.yield`, which may be left implicit.
  */
trait SingleBlockImplicitYieldTerminator extends Operation

trait AllElementTypesMatch(input: Attribute, output: Attribute) extends Operation