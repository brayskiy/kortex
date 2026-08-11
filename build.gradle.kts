plugins {
    kotlin("jvm") version "2.1.0"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))   // kotlin.test mapped onto JUnit 5
}

kotlin {
    jvmToolchain(21)
}

tasks.test {
    useJUnitPlatform()
    testLogging { events("passed", "skipped", "failed") }
}

application {
    // Top-level main() in Train.kt compiles to the class TrainKt.
    mainClass.set("TrainKt")
}

// Let `./gradlew run --args="..."` read stdin if ever needed.
tasks.named<JavaExec>("run") {
    standardInput = System.`in`
}
