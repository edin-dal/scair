// RUN: scala %s | filecheck %s

def main(args: Array[String]) = println("Hello, world!")

// CHECK: Hello, world!
