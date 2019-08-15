name := "spark-test"

version := "0.1"

scalaVersion := "2.12.5"

val sparkVersion = "2.4.3"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.9.9",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.9.9",
  "org.riversun" % "spark-ml-feature-importance-helper" % "1.0.0"
)





