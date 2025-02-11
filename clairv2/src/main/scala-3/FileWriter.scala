import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets

object TestTest {

  def main(args: Array[String]): Unit = {

    // Get the current working directory
    val currentDir = Paths.get(".").toAbsolutePath

    // Define the file path in the same directory
    val filePath = currentDir.resolve("output.txt")

    // Define the content to write
    val content = "Hello, this is a tes    t!"

    // Write content to the file
    Files.write(filePath, content.getBytes(StandardCharsets.UTF_8))

    println(s"File written to: ${filePath.toString}")
  }

}
