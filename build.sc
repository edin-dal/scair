import $ivy.`com.goyeau::mill-scalafix_mill0.11:0.5.0`
import com.goyeau.mill.scalafix.ScalafixModule

// import mill._, scalalib._
import mill.{Agg, RootModule, T, Task, PathRef, TaskModule}
import mill.scalalib.{ScalaModule, DepSyntax, scalafmt}
import mill.testrunner.TestResult
import mill.resolve.{Resolve, SelectMode}
import mill.define.{NamedTask, Command, ModuleRef}
import mill.main.Tasks
// import mill.eval.Evaluator
// import mill.util.Jvm
import scala.sys.process._
import scala.language.postfixOps
import java.io.PrintWriter

trait ScairSettings extends ScalaModule {
  def scalaVersion = "3.3.4"
  def scalacOptions = Seq("-Wunused:imports")
}

trait ScairModule extends ScairSettings with ScalafixModule {
  object test extends ScalaTests {

      def ivyDeps = Agg(ivy"org.scalatest::scalatest:3.2.19")
      def testFramework = "org.scalatest.tools.Framework"
    }
}

object `package` extends RootModule with ScairModule {

  def rootModule = ModuleRef(this)

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
  object clairV2 extends ScairModule {
    def moduleDeps = Seq(core)
  }

  object dialects extends ScairModule {
    def moduleDeps = Seq(clair, clairV2)
  }

  object gen_dialects extends ScairModule {
    def moduleDeps = Seq(dialects)

    override def generatedSources = T.sources {
      super.generatedSources() ++ generateDialects()
    }

    def oneDialect(src:String) = Task.Anon {
        dialects.runner().run(args = (os.Path(f"""${os.pwd}/out/${src.replace(".", "/")}_gen.scala""")).toString(), mainClass=src)
        print("Generated ")
        println(os.Path(f"""${os.pwd}/out/${src.replace(".", "/")}_gen.scala"""))
        PathRef(os.Path(f"""${os.pwd}/out/${src.replace(".", "/")}_gen.scala"""))
    }
        
    def generateDialects = {
          
      def dialectSource = Seq(
        "scair.dialects.affinegen.AffineGen",
        "scair.dialects.arithgen.ArithGen",
        "scair.dialects.cmathgen.CMathGen",
        "scair.dialects.funcgen.FuncGen",
        "scair.dialects.llvmgen.LLVMGen",
        "scair.dialects.memrefgen.MemrefGen",
        "scair.dialects.cmathv2.CMathV2Gen"
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

  def allScalaSources = Tasks(Resolve.Tasks.resolve(this, Seq("__.sources"), SelectMode.Multi).fold(sys.error(_), (tasks : List[NamedTask[_]]) => tasks.filter(_.isInstanceOf[NamedTask[Seq[PathRef]]]).map(_.asInstanceOf[NamedTask[Seq[PathRef]]])))
  def runAllUnitTests = T{T.sequence(Resolve.Tasks.resolve(this, Seq("__.test"), SelectMode.Multi).fold(sys.error(_), (tasks : List[NamedTask[_]]) => tasks.filter(_.isInstanceOf[NamedTask[(String, Seq[TestResult])]]).map(_.asInstanceOf[NamedTask[(String, Seq[TestResult])]])))()}

  // def fixAll = 
   
  def formatAll() = T.command {
    fix()
    scalafmt.ScalafmtModule.reformatAll(allScalaSources)
  }

  def checkFormatAll() = T.command {
    fix("--check")
    scalafmt.ScalafmtModule.checkFormatAll(allScalaSources)
  }

  def testAll() = T.command {
    runAllUnitTests()
    filechecks.run()()
  }


  object filechecks extends TaskModule {
    def classpath = Task.Anon {
      println("Storing full classpath for lit access")
      val full_cp = tools.transitiveCompileClasspath()
      
      val file = new PrintWriter((rootModule().millModuleBasePath.value / "full-classpath").toString())
      file.print(
        full_cp.map(_.path).mkString(":")
      )
      file.flush()
      file.close()
    }

    override def defaultCommandName(): String = "run"

    def run() = Task.Command {
        tools.launcher()
        filechecks.classpath()

        val rootPath = rootModule().millModuleBasePath.value
        val litStatus = os.proc("lit", "tests/filecheck", "-v").call(cwd = rootPath, propagateEnv = true, stdout = os.Inherit, stderr = os.Inherit)
        if (litStatus.exitCode != 0) {
          sys.error(f"Filechecks failed")

        }
        
      }
    
}

}
