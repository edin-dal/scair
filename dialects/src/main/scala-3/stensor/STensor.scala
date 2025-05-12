package scair.dialects.stensor

// import scair.AttrParser
import scair.ir.*
import fastparse.*

import scair.dialects.builtin.*
import scair.dialects.isl.*

import scair.clair.macros.*

import com.github.papychacal.isl as isl
case class STensor(
    val typ: Attribute,
    val dimensionList: ArrayAttribute[IntData],
    val known: Map,
    val redundant: Map
) extends ParametrizedAttribute(
      name = "stensor.stensor",
      parameters = Seq(typ, dimensionList, known, redundant)
    )
    with TypeAttribute
    with MLIRName["stensor.stensor"]
    derives AttributeTrait:

    override def custom_verify(): Unit =
        if known.map.range().tupleDim() != 1 then
            throw new Exception(
              s"The known map must be of dimension 1, got ${known.map.range().tupleDim()}"
            )
        if known.map.range().dimMaxVal(0).ne(known.map.range().dimMinVal(0)) != 0 then
            throw new Exception(
              s"The known map must map to single values, got the range [${known.map.range().dimMinVal(0)}, ${known.map.range().dimMaxVal(0)}]"
            )
        if known.map.domain().tupleDim() != dimensionList.size then
            throw new Exception(
              s"The known map domain must match dimensionality of the tensor, got ${known.map.domain().tupleDim()} instead of ${dimensionList.size}"
            )
        
        if redundant.map.domain().tupleDim() != dimensionList.size then
            throw new Exception(
              s"The redundant map domain must match dimensionality of the tensor, got ${redundant.map.domain().tupleDim()} instead of ${dimensionList.size}"
            )
        if redundant.map.range().tupleDim() != dimensionList.size then
            throw new Exception(
              s"The redundant map must match dimensionality of the tensor, got ${redundant.map.range().tupleDim()} instead of ${dimensionList.size}"
            )
    

val STensorDialect = summonDialect[Tuple1[STensor], EmptyTuple](Seq()) 


