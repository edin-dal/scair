package scair.ir

/** The outcome of visiting an operation during a walk.
  *
  *   - Advance: keep walking, including into the visited operation's regions.
  *   - Skip: keep walking, but do not descend into the visited operation's
  *     regions. Meaningless in post-order, where regions are already walked.
  *   - Interrupt: stop the whole walk.
  *
  * Walks are allocation-free per visited node: the callback is a single
  * function object for the whole walk, and containers are iterated inline. The
  * callback may erase the operation being visited (in pre-order, it must then
  * return Skip or Interrupt), but no other operation of the walk.
  */
enum WalkResult:
  case Advance, Skip, Interrupt
