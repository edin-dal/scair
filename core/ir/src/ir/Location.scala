package scair.ir

sealed trait Location

case object UnknownLoc extends Location

final case class FileLineColLoc(
    filename: String,
    line: Int,
    column: Int,
) extends Location

final case class FileLineColRange(
    filename: String,
    startLine: Int,
    startColumn: Int,
    endLine: Int,
    endColumn: Int,
) extends Location
