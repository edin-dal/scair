val scala_version = "3.3.1"

scalaVersion := scala_version

libraryDependencies ++= Seq(
  "com.lihaoyi" %% "fastparse" % "3.1.0",
  "org.scalatest" % "scalatest_3" % "3.2.19" % "test",
  "com.github.scopt" %% "scopt" % "4.1.0"
)

enablePlugins(JavaAppPackaging)
