package scair.clair

import scair.ir.*

import scala.compiletime.deferred

transparent trait DerivedAttribute[name <: String]
    extends ParametrizedAttribute:

  // This enables summoning the right instance without an explicit type parameter
  protected final given defs
      : AttrDefs[? >: this.type <: DerivedAttribute[name]] = deferred

  override def name: String = defs.name

  override def parameters: Seq[Attribute] =
    defs.parameters(this)
