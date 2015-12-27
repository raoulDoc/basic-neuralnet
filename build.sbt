name := "NeuralNet"

version := "1.0"

scalaVersion := "2.11.7"

resolvers +=
    "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"


libraryDependencies  ++= Seq(
    // other dependencies here
    "org.scalanlp" %% "breeze" % "0.10",
    // native libraries are not included by default. add this if you want them (as of 0.7)
    // native libraries greatly improve performance, but increase jar sizes.
    "org.scalanlp" %% "breeze-natives" % "0.10",
        "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")