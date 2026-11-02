package scair.passes.reconcile

import scair.MLContext
import scair.transformations.*

final case class ReconcileUnrealizedCasts() extends WalkerPass
    derives PassCompanion:
  override def walker(using MLContext) = ReconcileCasts.walker
