package scair.dialects.isl

import com.github.papychacal.isl as isl
import scair.ir.DataAttribute
import scair.ir.AttributeObject
import scair.AttrParser
import scair.ir.Attribute
import fastparse.*
import scair.clair.macros.summonDialect


given isl.Ctx = isl.Ctx()

object Map extends AttributeObject:
    override def parse[$: ParsingRun](p: AttrParser): ParsingRun[Attribute] = 
        given Whitespace = NoWhitespace.noWhitespaceImplicit
        "<"~ (CharIn("{[") ~ CharPred(_ != '}').rep ~ "}").! ~ ">" map(str => Map(isl.Map(str)))
    override def name: String = "isl.map"

case class Map(
    map: isl.Map
) extends DataAttribute[isl.Map](
      name = "isl.map",
      map
    )


object Set extends AttributeObject:
    override def parse[$: ParsingRun](p: AttrParser): ParsingRun[Attribute] = 
        given Whitespace = NoWhitespace.noWhitespaceImplicit
        "<"~ (CharIn("{[") ~ CharPred(_ != '}').rep ~ "}").! ~ ">" map(str => Set(isl.Set(str)))
    override def name: String = "isl.set"
case class Set(
    set: isl.Set
) extends DataAttribute[isl.Set](
      name = "isl.set",
      set
    )
    

val ISLDialect = summonDialect[EmptyTuple, EmptyTuple](Seq(Map, Set))