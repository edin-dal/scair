import mill._, scalalib._
import mill.util.Jvm


object scair extends RootModule with ScalaModule {
  def scalaVersion = "3.3.4"
  def semanticdbVersion = "4.4.30"
  def scalacOptions = Seq("-Wunused:imports")
  def ivyDeps = Agg(
    ivy"org.scalatest::scalatest:3.2.19",
    ivy"com.lihaoyi::fastparse:3.1.0",
    ivy"com.github.scopt::scopt:4.1.0"
  )
  def testFrameworks = Seq("org.scalatest.tools.Framework")

  object core extends ScalaModule {
    def scalaVersion = scair.scalaVersion
    def ivyDeps = scair.ivyDeps
  }

  object ScaIRDL extends ScalaModule {
    def scalaVersion = scair.scalaVersion
    def moduleDeps = Seq(core)
  }

  object clair extends ScalaModule {
    def scalaVersion = scair.scalaVersion
    def moduleDeps = Seq(ScaIRDL)
  }

  object dialects extends ScalaModule {
    def scalaVersion = scair.scalaVersion
    def moduleDeps = Seq(clair)
  }

  def one_dialect() = {
    // T {
    
  }

  object gen_dialects extends ScalaModule {
    def scalaVersion = scair.scalaVersion
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

  object transformations extends ScalaModule {
    def scalaVersion = scair.scalaVersion
    def moduleDeps = Seq(gen_dialects)
  }

  object tools extends ScalaModule with JavaModule {
    def scalaVersion = scair.scalaVersion
    def moduleDeps = Seq(gen_dialects, transformations)
    def mainClass = Some("ScairOpt")
  }
}
