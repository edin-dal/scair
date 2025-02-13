import mill._, scalalib._
import mill.util.Jvm
import scala.sys.process._
import scala.language.postfixOps
import java.io.PrintWriter

trait ScairSettings extends ScalaModule {
  def scalaVersion = "3.3.4"
  def semanticdbVersion = "4.4.30"
  def scalacOptions = Seq("-Wunused:imports")
}

trait ScairModule extends ScairSettings {
  object test extends ScalaTests {

      def ivyDeps = Agg(ivy"org.scalatest::scalatest:3.2.19")
      def testFramework = "org.scalatest.tools.Framework"
    }
}

object scair extends RootModule with ScairModule {

  object core extends ScairModule {

    def ivyDeps = Agg(
      ivy"com.lihaoyi::fastparse:3.1.0",
      ivy"com.github.scopt::scopt:4.1.0"
    )
  }

  object ScaIRDL extends ScairModule {
    def moduleDeps = Seq(core)
  }

  object clair extends ScairModule {
    def moduleDeps = Seq(ScaIRDL)
  }

  object dialects extends ScairModule {
    def moduleDeps = Seq(clair)
  }

  def one_dialect() = {
    // T {
    
  }

  object gen_dialects extends ScairModule {
    def moduleDeps = Seq(dialects)

    override def sources = T.sources {
      super.sources() ++ generateDialects()
    }

    def oneDialect(src:String) = T.command {
        dialects.runMain(mainClass=src, args = (os.Path(f"""${os.pwd}/out/${src.replace(".", "/")}_gen.scala""")).toString())()
        // throw new Exception(genFile.toString())
        PathRef(os.Path(f"""${os.pwd}/out/${src.replace(".", "/")}_gen.scala"""))
    }
        
    def generateDialects = {
          
      def dialectSource = Seq(
        "scair.dialects.affinegen.AffineGen",
        "scair.dialects.arithgen.ArithGen",
        "scair.dialects.cmathgen.CMathGen",
        "scair.dialects.funcgen.FuncGen",
        "scair.dialects.llvmgen.LLVMGen",
        "scair.dialects.memrefgen.MemrefGen"
      ) 

      T.traverse(dialectSource)(oneDialect)
    }
  }

  object transformations extends ScairModule {
    def moduleDeps = Seq(gen_dialects)
  }

  object tools extends ScairModule  {
    def moduleDeps = Seq(gen_dialects, transformations)
    def mainClass = Some("ScairOpt")
  }

  object filechecks extends ScairSettings {
    def classpath = T {
      val full_cp = tools.compileClasspath()
      val file = new PrintWriter("full-classpath")
      file.print(
        full_cp.map(_.path).mkString(":")
      )
      file.flush()
      file.close()
    }

    def moduleDeps = Seq(tools)

    def run = T {
      test.run()
    }
    
    object test extends TaskModule {
      def defaultCommandName = "run"
      def run = T.input {
        tools.launcher()
        filechecks.classpath()

        val r = ("lit tests/filecheck -v" !)
        if (r != 0) {
          sys.error("Filechecks failed")
        }
        
      }
  }
}

}
