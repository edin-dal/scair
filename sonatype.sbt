import xerial.sbt.Sonatype._

publishMavenStyle := true

sonatypeProfileName := "io.github.edin-dal"

sonatypeCredentialHost := sonatypeCentralHost

homepage := Some(url("https://github.com/edin-dal/scair/"))
scmInfo := Some(
  ScmInfo(
    url("https://github.com/edin-dal/scair/"),
    "scm:git@github.com:edin-dal/scair.git"
  )
)

// credentials += Credentials(
//   "Sonatype Nexus Repository Manager",
//   "central.sonatype.com",
//   System.getenv("SONATYPE_USERNAME"), // GitHub username from environment variable
//   System.getenv("SONATYPE_PASSWORD") // GitHub token from environment variable
// )

developers := List(
  Developer(id="baymaks", name="Maks Kret", email="maksymilian.kret@ed.ac.uk", url=url("https://github.com/baymaks"))
)

licenses := Seq("APL2" -> url("https://github.com/edin-dal/scair/blob/main/LICENSE"))

publishTo := sonatypePublishToBundle.value
