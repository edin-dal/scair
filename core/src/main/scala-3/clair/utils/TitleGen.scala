package scair.clair

import scair.ir.ListType

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

object SubTitleGen:

  def generate_sub(subtitle: String): String =
    val len = subtitle.length
    assert(len > 0)

    val top = s"/*≡≡=--=≡${"≡" * len}≡=--=≡≡*\\ \n"
    val mid = s"||       ${subtitle.toUpperCase()}       ||\n"
    val bot = s"\\*≡==---=${"≡" * len}=---==≡*/\n"

    return top + mid + bot

object TitleGen:

  /** Generate a title in ascii art.
    *
    * @param title
    * @return
    *   ascii art title
    */
  def generate(title: String): String =

    val lower_title = title.toLowerCase().split("\\s+")
    val title_size = lower_title.size
    val letter_size = Letters.tuple_length
    val lines =
      title_size * letter_size +
        2 // + title_size + 1 to create horizontal \n lines in between words
    val charMap = Letters.charMap

    val generated_title = ListType.fill(lines)(new StringBuilder(""))
    var index = 0

    generated_title(0) += '\n'

    for word <- lower_title do
      for letter <- word do
        val letter_tup = charMap(letter)
        for i <- 0 to letter_size - 1 do
          generated_title(i + (index * letter_size) + 1) ++= letter_tup(i) + " "
      for i <- 0 to letter_size - 1 do
        generated_title(i + (index * letter_size) + 1) += '\n'
      index += 1
      generated_title(index * letter_size + 1) += '\n'

    val final_string =
      val finale = new StringBuilder("")
      generated_title.map((x: StringBuilder) => finale ++= x)
      finale.result()
    final_string

  /** CLI hook for generating a title in ascii art. Takes in string arguments
    * and generates a title. If no arguments are given, generates a default
    * title.
    *
    * To run this function, use the following command with mill: `./mill
    * clair.run your string of your choice`
    *
    * @param args
    */
  def main(args: Array[String]): Unit =
    if args.length == 0 then print(generate("title generator and whatnot"))
    else print(generate(args.mkString(" ")))

object Letters:

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
      "░╚═════╝░",
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
      "╚═╝░░░░░╚═╝",
    )

  val letterN: Seq[String] =
    Seq(
      "███╗░░██╗",
      "████╗░██║",
      "██╔██╗██║",
      "██║╚████║",
      "██║░╚███║",
      "╚═╝░░╚══╝",
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
      "░░░╚═╝░░░",
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
      "░░░╚═╝░░░",
    )

  val letterU: Seq[String] =
    Seq(
      "██╗░░░██╗",
      "██║░░░██║",
      "██║░░░██║",
      "██║░░░██║",
      "╚██████╔╝",
      "░╚═════╝░",
    )

  val letterV: Seq[String] =
    Seq(
      "██╗░░░██╗",
      "██║░░░██║",
      "╚██╗░██╔╝",
      "░╚████╔╝░",
      "░░╚██╔╝░░",
      "░░░╚═╝░░░",
    )

  val letterW: Seq[String] =
    Seq(
      "██╗░░░░░░░██╗",
      "██║░░██╗░░██║",
      "╚██╗████╗██╔╝",
      "░████╔═████║░",
      "░╚██╔╝░╚██╔╝░",
      "░░╚═╝░░░╚═╝░░",
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
      "░░░╚═╝░░░",
    )

  val letterZ: Seq[String] =
    Seq("███████╗", "╚════██║", "░░███╔═╝", "██╔══╝░░", "███████╗", "╚══════╝")

  val number1: Seq[String] =
    Seq("░░███╗░░", "░████║░░", "██╔██║░░", "╚═╝██║░░", "███████╗", "╚══════╝")

  val number2: Seq[String] =
    Seq("██████╗░", "╚════██╗", "░░███╔═╝", "██╔══╝░░", "███████╗", "╚══════╝")

  val number3: Seq[String] =
    Seq("██████╗░", "╚════██╗", "░█████╔╝", "░╚═══██╗", "██████╔╝", "╚═════╝░")

  val number4: Seq[String] =
    Seq("░░██╗██╗", "░██╔╝██║", "██╔╝░██║", "███████║", "╚════██║", "░░░░░╚═╝")

  val number5: Seq[String] =
    Seq("███████╗", "██╔════╝", "██████╗░", "╚════██╗", "██████╔╝", "╚═════╝░")

  val number6: Seq[String] =
    Seq("░█████╗░", "██╔═══╝░", "██████╗░", "██╔══██╗", "╚█████╔╝", "░╚════╝░")

  val number7: Seq[String] =
    Seq("███████╗", "╚════██║", "░░░░██╔╝", "░░░██╔╝░", "░░██╔╝░░", "░░╚═╝░░░")

  val number8: Seq[String] =
    Seq("░█████╗░", "██╔══██╗", "╚█████╔╝", "██╔══██╗", "╚█████╔╝", "░╚════╝░")

  val number9: Seq[String] =
    Seq("░██████╗", "██╔══██╗", "╚██████║", "░╚═══██║", "░█████╔╝", "░╚════╝░")

  val charMap: Map[Char, Seq[String]] = Map(
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
    ('z' -> letterZ),
    ('1' -> number1),
    ('2' -> number2),
    ('3' -> number3),
    ('4' -> number4),
    ('5' -> number5),
    ('6' -> number6),
    ('7' -> number7),
    ('8' -> number8),
    ('9' -> number9),
  )
