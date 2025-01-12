package scair.clair

import scair.scairdl.irdef.ListType

// ████████╗ ██╗ ████████╗ ██╗░░░░░ ███████╗
// ╚══██╔══╝ ██║ ╚══██╔══╝ ██║░░░░░ ██╔════╝
// ░░░██║░░░ ██║ ░░░██║░░░ ██║░░░░░ █████╗░░
// ░░░██║░░░ ██║ ░░░██║░░░ ██║░░░░░ ██╔══╝░░
// ░░░██║░░░ ██║ ░░░██║░░░ ███████╗ ███████╗
// ░░░╚═╝░░░ ╚═╝ ░░░╚═╝░░░ ╚══════╝ ╚══════╝

// ░██████╗░ ███████╗ ███╗░░██╗ ███████╗ ██████╗░ ░█████╗░ ████████╗ ░█████╗░ ██████╗░
// ██╔════╝░ ██╔════╝ ████╗░██║ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██╔══██╗ ██╔══██╗
// ██║░░██╗░ █████╗░░ ██╔██╗██║ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║░░██║ ██████╔╝
// ██║░░╚██╗ ██╔══╝░░ ██║╚████║ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║░░██║ ██╔══██╗
// ╚██████╔╝ ███████╗ ██║░╚███║ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ╚█████╔╝ ██║░░██║
// ░╚═════╝░ ╚══════╝ ╚═╝░░╚══╝ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ░╚════╝░ ╚═╝░░╚═╝

object SubTitleGen {
  def generate_sub(subtitle: String): String = {
    val len = subtitle.length
    assert(len > 0)

    val top = s"/*≡≡=--=≡${"≡" * len}≡=--=≡≡*\\ \n"
    val mid = s"||       ${subtitle.toUpperCase()}       ||\n"
    val bot = s"\\*≡==---=${"≡" * len}=---==≡*/\n"

    return top + mid + bot
  }
}

object TitleGen {

  /** Generate a title in ascii art.
    *
    * @param title
    * @return ascii art title
    */
  def generate(title: String): String = {

    val lower_title = title.toLowerCase().split("\\s+")
    val title_size = lower_title.size
    val letter_size = Letters.tuple_length
    val lines =
      title_size * letter_size + 2 // + title_size + 1 to create horizontal \n lines in between words
    val letter_map = Letters.letter_map

    val generated_title = ListType.fill(lines) { new StringBuilder("") }
    var index = 0

    generated_title(0) += '\n'

    for (word <- lower_title) {
      for (letter <- word) {
        val letter_tup = letter_map(letter)
        for (i <- 0 to letter_size - 1) {
          generated_title(i + (index * letter_size) + 1) ++= letter_tup(i) + " "
        }
      }
      for (i <- 0 to letter_size - 1) {
        generated_title(i + (index * letter_size) + 1) += '\n'
      }
      index += 1
      generated_title(index * letter_size + 1) += '\n'
    }

    val final_string = {
      val finale = new StringBuilder("")
      generated_title.map((x: StringBuilder) => finale ++= x)
      finale.result()
    }
    final_string
  }

  /** CLI hook for generating a title in ascii art.
    * Takes in string arguments and generates a title.
    * If no arguments are given, generates a default title.
    * 
    * To run this function, use the following command with sbt:
    * `sbt "run/clair your string of your choice"`
    * 
    * @param args
    */
  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      print(generate("title generator and whatnot"))
    } else {
      print(generate(args.mkString(" ")))
    }
  }
}

object Letters {

  val tuple_length = 6

  val letterA: Seq[String] =
    Seq("░█████╗░", "██╔══██╗", "███████║", "██╔══██║", "██║░░██║", "╚═╝░░╚═╝")
  val letterB: Seq[String] =
    Seq("██████╗░", "██╔══██╗", "██████╦╝", "██╔══██╗", "██████╦╝", "╚═════╝░")
  val letterC: Seq[String] =
    Seq("░█████╗░", "██╔══██╗", "██║░░╚═╝", "██║░░██╗", "╚█████╔╝", "░╚════╝░")
  val letterD: Seq[String] =
    Seq("██████╗░", "██╔══██╗", "██║░░██║", "██║░░██║", "██████╔╝", "╚═════╝░")
  val letterE: Seq[String] =
    Seq("███████╗", "██╔════╝", "█████╗░░", "██╔══╝░░", "███████╗", "╚══════╝")
  val letterF: Seq[String] =
    Seq("███████╗", "██╔════╝", "█████╗░░", "██╔══╝░░", "██║░░░░░", "╚═╝░░░░░")
  val letterG: Seq[String] =
    Seq(
      "░██████╗░",
      "██╔════╝░",
      "██║░░██╗░",
      "██║░░╚██╗",
      "╚██████╔╝",
      "░╚═════╝░"
    )
  val letterH: Seq[String] =
    Seq("██╗░░██╗", "██║░░██║", "███████║", "██╔══██║", "██║░░██║", "╚═╝░░╚═╝")
  val letterI: Seq[String] =
    Seq("██╗", "██║", "██║", "██║", "██║", "╚═╝")
  val letterJ: Seq[String] =
    Seq("░░░░░██╗", "░░░░░██║", "░░░░░██║", "██╗░░██║", "╚█████╔╝", "░╚════╝░")
  val letterK: Seq[String] =
    Seq("██╗░░██╗", "██║░██╔╝", "█████═╝░", "██╔═██╗░", "██║░╚██╗", "╚═╝░░╚═╝")
  val letterL: Seq[String] =
    Seq("██╗░░░░░", "██║░░░░░", "██║░░░░░", "██║░░░░░", "███████╗", "╚══════╝")
  val letterM: Seq[String] =
    Seq(
      "███╗░░░███╗",
      "████╗░████║",
      "██╔████╔██║",
      "██║╚██╔╝██║",
      "██║░╚═╝░██║",
      "╚═╝░░░░░╚═╝"
    )
  val letterN: Seq[String] =
    Seq(
      "███╗░░██╗",
      "████╗░██║",
      "██╔██╗██║",
      "██║╚████║",
      "██║░╚███║",
      "╚═╝░░╚══╝"
    )
  val letterO: Seq[String] =
    Seq("░█████╗░", "██╔══██╗", "██║░░██║", "██║░░██║", "╚█████╔╝", "░╚════╝░")
  val letterP: Seq[String] =
    Seq("██████╗░", "██╔══██╗", "██████╔╝", "██╔═══╝░", "██║░░░░░", "╚═╝░░░░░")
  val letterQ: Seq[String] =
    Seq(
      "░██████╗░",
      "██╔═══██╗",
      "██║██╗██║",
      "╚██████╔╝",
      "░╚═██╔═╝░",
      "░░░╚═╝░░░"
    )
  val letterR: Seq[String] =
    Seq("██████╗░", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║░░██║", "╚═╝░░╚═╝")
  val letterS: Seq[String] =
    Seq("░██████╗", "██╔════╝", "╚█████╗░", "░╚═══██╗", "██████╔╝", "╚═════╝░")
  val letterT: Seq[String] =
    Seq(
      "████████╗",
      "╚══██╔══╝",
      "░░░██║░░░",
      "░░░██║░░░",
      "░░░██║░░░",
      "░░░╚═╝░░░"
    )
  val letterU: Seq[String] =
    Seq(
      "██╗░░░██╗",
      "██║░░░██║",
      "██║░░░██║",
      "██║░░░██║",
      "╚██████╔╝",
      "░╚═════╝░"
    )
  val letterV: Seq[String] =
    Seq(
      "██╗░░░██╗",
      "██║░░░██║",
      "╚██╗░██╔╝",
      "░╚████╔╝░",
      "░░╚██╔╝░░",
      "░░░╚═╝░░░"
    )
  val letterW: Seq[String] =
    Seq(
      "██╗░░░░░░░██╗",
      "██║░░██╗░░██║",
      "╚██╗████╗██╔╝",
      "░████╔═████║░",
      "░╚██╔╝░╚██╔╝░",
      "░░╚═╝░░░╚═╝░░"
    )
  val letterX: Seq[String] =
    Seq("██╗░░██╗", "╚██╗██╔╝", "░╚███╔╝░", "░██╔██╗░", "██╔╝╚██╗", "╚═╝░░╚═╝")
  val letterY: Seq[String] =
    Seq(
      "██╗░░░██╗",
      "╚██╗░██╔╝",
      "░╚████╔╝░",
      "░░╚██╔╝░░",
      "░░░██║░░░",
      "░░░╚═╝░░░"
    )
  val letterZ: Seq[String] =
    Seq("███████╗", "╚════██║", "░░███╔═╝", "██╔══╝░░", "███████╗", "╚══════╝")

  val letter_map: Map[Char, Seq[String]] = Map(
    ('a' -> letterA),
    ('b' -> letterB),
    ('c' -> letterC),
    ('d' -> letterD),
    ('e' -> letterE),
    ('f' -> letterF),
    ('g' -> letterG),
    ('h' -> letterH),
    ('i' -> letterI),
    ('j' -> letterJ),
    ('k' -> letterK),
    ('l' -> letterL),
    ('m' -> letterM),
    ('n' -> letterN),
    ('o' -> letterO),
    ('p' -> letterP),
    ('q' -> letterQ),
    ('r' -> letterR),
    ('s' -> letterS),
    ('t' -> letterT),
    ('u' -> letterU),
    ('v' -> letterV),
    ('w' -> letterW),
    ('x' -> letterX),
    ('y' -> letterY),
    ('z' -> letterZ)
  )
}
