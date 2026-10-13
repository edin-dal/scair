package scair.passes.reconcile

import scair.MLContext
import scair.transformations.*

final class ReconcileUnrealizedCasts(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "reconcile-unrealized-casts"

  override final val walker = ReconcileCasts.walker
