/**
 * This Gradle build script configures the 'ml' module, which encapsulates all
 * machine learning functionalities for the FGO-Bot application.
 *
 * It is responsible for managing dependencies related to computer vision and
 * on-device inference, primarily through TensorFlow Lite. The module is designed
 * to be a self-contained unit, providing ML-driven insights and decisions to the
 * main 'app' module while remaining decoupled from its core logic.
 */
plugins {
    id("com.android.library")
    id("kotlin-android")
}

android {
    namespace = "io.github.fate_grand_automata.ml"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        mlModelBinding = true
    }
}

dependencies {
    // TensorFlow Lite for on-device ML model inference
    implementation("org.tensorflow:tensorflow-lite:2.12.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.12.0") // For GPU acceleration
    implementation("org.tensorflow:tensorflow-lite-support:0.4.3")

    // Core Android and Kotlin libraries
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Testing libraries
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
} 