package scair.dialects.isl

import com.github.papychacal.isl.*
import scair.ir.DataAttribute
import scair.ir.AttributeObject
import scair.AttrParser
import scair.ir.Attribute
import fastparse.*
import scair.clair.macros.summonDialect

object Map extends AttributeObject:
    override def parse[$: ParsingRun](p: AttrParser): ParsingRun[Attribute] = 
        given Whitespace = NoWhitespace.noWhitespaceImplicit
        given Ctx = Ctx()
        "<"~ ("{" ~ CharPred(_ != '}').rep ~ "}").!.? ~ ">" map { parsed_map =>
            parsed_map match
                case Some(str) =>
            
                    val map = BasicMap(str)
                    Map(map)
                case None => 
                    Map(BasicMap("{[i] -> [j]}"))
        }
    override def name: String = "isl.basic_map"

case class Map(
    map: BasicMap
) extends DataAttribute[BasicMap](
      name = "isl.basic_map",
      map
    ):

  override def custom_print: String = 
    val p = Printer(using Ctx())()
    p.printBasicMap(map)
    s"${prefix}${name}<${p.getStr()}>"

val ISLDialect = summonDialect[EmptyTuple, EmptyTuple](Seq(Map))