package main.eenum.macros

import scala.annotation.switch
import scala.quoted.*

inline def what[T: Type](): Unit = ()