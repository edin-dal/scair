// ITERATION 1

// trait A(b: Int) {}
// trait B
// trait C

// class MyClass extends A(5) with B with C

// object ReflectionExample {
//   def main(args: Array[String]): Unit = {
//     // Get the class of MyClass
//     val clazz = classOf[MyClass]

//     // Get the interfaces (traits) that MyClass implements
//     val interfaces = clazz.getInterfaces

//     // Print out the names of the interfaces (traits)
//     interfaces.foreach(interface => println(interface.getName))
//   }
// }

// ITERATION 2

// import java.lang.reflect.{Constructor, Field}

// trait A(b: Int)
// trait B
// trait C

// class MyClass(e1: Int, e2: Int) extends A(5) with B with C

// object ReflectionExample {
//   def main(args: Array[String]): Unit = {
//     // Get the class of MyClass
//     val clazz = classOf[MyClass]

//     // Get the constructor of the class
//     val constructor: Constructor[_] = clazz.getDeclaredConstructor()

//     // Make the constructor accessible if it is private or protected
//     constructor.setAccessible(true)

//     // Create an instance of MyClass using the constructor
//     val instance = constructor.newInstance()

//     // Get all fields of the class (including inherited fields)
//     val fields: Array[Field] = clazz.getDeclaredFields

//     // Print the field names and their values
//     println("Fields in MyClass:")
//     fields.foreach { field =>
//       field.setAccessible(true) // Make the field accessible if it is private
//       println(
//         s"Field: ${field.getName}, Type: ${field.getType}, Value: ${field.get(instance)}"
//       )
//     }
//   }
// }

// ITERATION 3

// import java.lang.reflect.{Constructor, Field}

// trait A(b: Int)
// trait B
// trait C

// // MyClass has a constructor with parameters e1 and e2
// class MyClass(e1: Int, e2: Int) extends A(5) with B with C

// object ReflectionExample {
//   def main(args: Array[String]): Unit = {
//     // Get the class of MyClass
//     val clazz = classOf[MyClass]

//     // Get the constructor that takes two Int parameters (e1, e2)
//     val constructor: Constructor[_] =
//       clazz.getDeclaredConstructor(classOf[Int], classOf[Int])

//     // Make the constructor accessible if it is private or protected
//     constructor.setAccessible(true)

//     // Create an instance of MyClass using the constructor and passing values for e1 and e2
//     val instance =
//       constructor.newInstance(10.asInstanceOf[Object], 20.asInstanceOf[Object])

//     // Get all fields of the class (including inherited fields from traits)
//     val fields: Array[Field] = clazz.getDeclaredFields

//     // Print the field names and their values
//     println("Fields in MyClass:")
//     fields.foreach { field =>
//       field.setAccessible(true) // Make the field accessible if it is private
//       println(
//         s"Field: ${field.getName}, Type: ${field.getType}, Value: ${field.get(instance)}"
//       )
//     }
//   }
// }

// ITERATION 4

// import java.lang.reflect.{Constructor, Field}
// import scala.reflect.ClassTag

// trait A(b: Int)
// trait B
// trait C

// // MyClass has a constructor with two parameters e1 and e2
// case class MyClass(e1: Int, e2: Int, e3: String = "bruh")
//     extends A(5)
//     with B
//     with C

// object ReflectionExample {

//   def reflectionExample[T <: MyClass: ClassTag](): Unit = {

//     // Get the class of the type T using ClassTag
//     val clazz = implicitly[ClassTag[T]].runtimeClass

//     // Get the class of MyClass
//     // val clazz = classOf[T]

//     // Get all the constructors of the class
//     val constructors: Array[Constructor[_]] = clazz.getDeclaredConstructors

//     // Choose the first constructor (or you can loop through all constructors if needed)
//     val constructor = constructors(0) // Take the first constructor dynamically

//     // Make the constructor accessible if it is private or protected
//     constructor.setAccessible(true)

//     // Dynamically retrieve the parameter types of the constructor
//     val paramTypes = constructor.getParameterTypes

//     // Dynamically create the arguments for the constructor based on parameter types
//     val dynamicArgs = paramTypes.map {
//       case t if t == classOf[Int] =>
//         10.asInstanceOf[Object] // Assigning a value for Int
//       case t if t == classOf[String] =>
//         "example".asInstanceOf[Object] // Assigning a value for String
//       case _ =>
//         null: Object // If other types are needed, you can add additional cases
//     }

//     // Create an instance of MyClass using the constructor with dynamic arguments
//     val instance = constructor.newInstance(dynamicArgs: _*)

//     // Get all fields of the class (including inherited fields)
//     val fields: Array[Field] = clazz.getDeclaredFields

//     // Print the field names and their values
//     println("Fields in MyClass:")
//     fields.foreach { field =>
//       field.setAccessible(true) // Make the field accessible if it is private
//       println(
//         s"Field: ${field.getName}, Type: ${field.getType}, Value: ${field.get(instance)}"
//       )
//     }
//   }

//   def main(args: Array[String]): Unit = {
//     reflectionExample[MyClass]()
//   }
// }

// ITERATION 5
// import java.lang.reflect.{Constructor, Field}
// import scala.reflect.ClassTag

// trait A(b: Int)
// trait B
// trait C

// // MyClass has a constructor with two parameters e1, e2 and a default for e3
// case class MyClass(e1: Int, e2: Int, e3: String = "bruh")
//     extends A(5)
//     with B
//     with C

// object ReflectionExample {

//   def reflectionExample[T <: MyClass: ClassTag](): Unit = {

//     // Get the class of the type T using ClassTag
//     val clazz = implicitly[ClassTag[T]].runtimeClass

//     // Get all the constructors of the class
//     val constructors: Array[Constructor[_]] = clazz.getDeclaredConstructors

//     // Choose the first constructor (or you can loop through all constructors if needed)
//     val constructor = constructors(0) // Take the first constructor dynamically

//     // Make the constructor accessible if it is private or protected
//     constructor.setAccessible(true)

//     // Dynamically retrieve the parameter types of the constructor
//     val paramTypes = constructor.getParameterTypes

//     // Dynamically create the arguments for the constructor based on parameter types
//     val dynamicArgs = paramTypes.zipWithIndex.map {
//       case (t, idx) if t == classOf[Int] =>
//         // Assigning a value for Int parameters
//         idx match {
//           case 0 => 10.asInstanceOf[Object] // First parameter e1
//           case 1 => 20.asInstanceOf[Object] // Second parameter e2
//           case _ => null
//         }
//       case (t, idx) if t == classOf[String] =>
//         // If String parameter exists, check if it has a default value
//         if (idx == 2) {
//           null.asInstanceOf[Object] // We leave out the default parameter value (e3), it will use the default
//         } else {
//           "example"
//             .asInstanceOf[Object] // Assigning a value for String parameter
//         }
//       case _ =>
//         null: Object // If other types are needed, you can add additional cases
//     }

//     // Create an instance of MyClass using the constructor with dynamic arguments
//     val instance = constructor.newInstance(dynamicArgs: _*)

//     // Get all fields of the class (including inherited fields)
//     val fields: Array[Field] = clazz.getDeclaredFields

//     // Print the field names and their values
//     println("Fields in MyClass:")
//     fields.foreach { field =>
//       field.setAccessible(true) // Make the field accessible if it is private
//       println(
//         s"Field: ${field.getName}, Type: ${field.getType}, Value: ${field.get(instance)}"
//       )
//     }
//   }

//   def main(args: Array[String]): Unit = {
//     reflectionExample[MyClass]()
//   }
// }

// ITERATION 6
import java.lang.reflect.{Constructor, Field}
import scala.reflect.ClassTag

trait A(b: Int)
trait B
trait C

// MyClass with default value for e3
case class MyClass(e1: Int, e2: Int = 5, e3: String) extends A(5) with B with C

object MyClass {
  // Companion object where you store default parameter values
  val defaultValues: Map[String, Any] = Map(
    "e3" -> "bruh"
  )
}

object ReflectionExample {

  def reflectionExample[T <: MyClass: ClassTag](): Unit = {

    // Get the class of the type T using ClassTag
    val clazz = implicitly[ClassTag[T]].runtimeClass

    // Get all the constructors of the class
    val constructors: Array[Constructor[_]] = clazz.getDeclaredConstructors

    // Choose the first constructor (or you can loop through all constructors if needed)
    val constructor = constructors(0) // Take the first constructor dynamically

    // Make the constructor accessible if it is private or protected
    constructor.setAccessible(true)

    // Dynamically retrieve the parameter types of the constructor
    val paramTypes = constructor.getParameterTypes

    // Dynamically create the arguments for the constructor based on parameter types
    val dynamicArgs = paramTypes.zipWithIndex.map {
      case (t, idx) if t == classOf[Int] =>
        // Assigning a value for Int parameters
        idx match {
          case 0 => 10.asInstanceOf[Object] // First parameter e1
          case 1 => 20.asInstanceOf[Object] // Second parameter e2
          case _ => null
        }
      case (t, idx) if t == classOf[String] =>
        // Handle the default value for String type if it's not passed explicitly
        // Check if the parameter is part of the default values
        MyClass.defaultValues
          .get("e3")
          .getOrElse("default")
          .asInstanceOf[
            Object
          ] // Using the default for e3 (from companion object)
      case _ =>
        null: Object // If other types are needed, you can add additional cases
    }

    // Create an instance of MyClass using the constructor with dynamic arguments
    val instance = constructor.newInstance(dynamicArgs: _*)

    val interfaces = clazz.getInterfaces()

    // Get all fields of the class (including inherited fields)
    val fields: Array[Field] = clazz.getDeclaredFields

    // Print the field names and their values
    println("Fields in MyClass:")
    fields.foreach { field =>
      field.setAccessible(true) // Make the field accessible if it is private
      println(
        s"Field: ${field.getName}, Type: ${field.getType}, Value: ${field.get(instance)}"
      )
    }

    // interface.getName() is a string
    interfaces.foreach { interface => println(interface.getName()) }
  }

  def main(args: Array[String]): Unit = {
    reflectionExample[MyClass]()
  }
}
