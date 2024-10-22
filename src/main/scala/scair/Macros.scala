package scair
import scala.annotation.MacroAnnotation
import scala.annotation.experimental
import scala.quoted.Quotes
import scala.annotation.compileTimeOnly

@experimental
class PseudoMacro extends MacroAnnotation {
  override def transform(using quotes: Quotes)(
      tree: quotes.reflect.Definition
  ): List[quotes.reflect.Definition] = {
    import quotes.reflect.report
    report.warning("Warning, this is not a real macro")
    return List(tree)
  }
}
