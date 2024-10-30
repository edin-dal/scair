// RUN: scala -classpath ../../../target/scala-3.3.1/classes/ %s | filecheck %s

package scair

import scair.clair.{TitleGen, SubTitleGen}

def main(args: Array[String]) = {
  println(TitleGen.generate("Hello World"))
  println(SubTitleGen.generate_sub("Hello again, World!"))
}

// CHECK:       ██╗░░██╗ ███████╗ ██╗░░░░░ ██╗░░░░░ ░█████╗░
// CHECK-NEXT:  ██║░░██║ ██╔════╝ ██║░░░░░ ██║░░░░░ ██╔══██╗
// CHECK-NEXT:  ███████║ █████╗░░ ██║░░░░░ ██║░░░░░ ██║░░██║
// CHECK-NEXT:  ██╔══██║ ██╔══╝░░ ██║░░░░░ ██║░░░░░ ██║░░██║
// CHECK-NEXT:  ██║░░██║ ███████╗ ███████╗ ███████╗ ╚█████╔╝
// CHECK-NEXT:  ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ╚══════╝ ░╚════╝░
// CHECK-EMPTY:
// CHECK-NEXT:  ██╗░░░░░░░██╗ ░█████╗░ ██████╗░ ██╗░░░░░ ██████╗░
// CHECK-NEXT:  ██║░░██╗░░██║ ██╔══██╗ ██╔══██╗ ██║░░░░░ ██╔══██╗
// CHECK-NEXT:  ╚██╗████╗██╔╝ ██║░░██║ ██████╔╝ ██║░░░░░ ██║░░██║
// CHECK-NEXT:  ░████╔═████║░ ██║░░██║ ██╔══██╗ ██║░░░░░ ██║░░██║
// CHECK-NEXT:  ░╚██╔╝░╚██╔╝░ ╚█████╔╝ ██║░░██║ ███████╗ ██████╔╝
// CHECK-NEXT:  ░░╚═╝░░░╚═╝░░ ░╚════╝░ ╚═╝░░╚═╝ ╚══════╝ ╚═════╝░
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:  /*≡≡=--=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=--=≡≡*\
// CHECK-NEXT:  ||       HELLO AGAIN, WORLD!       ||
// CHECK-NEXT:  \*≡==---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---==≡*/
