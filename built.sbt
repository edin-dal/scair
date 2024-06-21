val scala_version = "2.12.13"

scalaVersion := scala_version

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-reflect" % scala_version,
  "com.lihaoyi" %% "fastparse" % "2.2.2",
  "org.scalatest" % "scalatest_2.12" % "3.1.1" % "test",
  "com.github.scopt" %% "scopt" % "4.1.0"
)

enablePlugins(JavaAppPackaging)
