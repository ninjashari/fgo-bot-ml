# FGA: Machine Learning Enhancement Action Plan

This document outlines a comprehensive, phased action plan for integrating a machine learning-based decision engine into the Fate/Grand Automata (FGA) application. The core principle of this plan is **modularity**—to build the new feature as a distinct, opt-in "ML Mode" without altering the existing, stable scripting functionality.

## High-Level Goal

The objective is to develop an "ML Mode" that enables FGA to:

1.  **Perceive:** Intelligently understand the complete battle state using a modern machine learning model for screen recognition.
2.  **Decide:** Determine the optimal course of action using a pre-trained Reinforcement Learning (RL) agent.
3.  **Act:** Execute the decided actions by leveraging the existing `libautomata` infrastructure for clicks and swipes.
4.  **Learn (Advanced):** Optionally collect gameplay data to allow for future on-device fine-tuning of the decision-making model.

The architecture will be composed of new, independent Gradle modules to ensure a clean separation of concerns.

---

## Phase 1: The "Eyes" - `ml-recognizer` Module

This phase focuses on replacing the current template-matching system with a robust object detection model for superior screen analysis.

-   **Action 1.1: Data Collection & Annotation**
    -   **Task:** Gather thousands of diverse FGO screenshots covering a wide variety of servants, CEs, enemies, items, and UI states.
    -   **Tooling:** Use a dedicated annotation tool (e.g., CVAT, LabelImg) to create a high-quality dataset by drawing bounding boxes around all relevant game entities. This is the foundation of the entire system.

-   **Action 1.2: Train an Object Detection Model**
    -   **Task:** Train a lightweight, high-performance object detection model suitable for on-device inference.
    -   **Model Choice:** A model like **YOLOv8** or **EfficientDet-Lite** is recommended for its balance of speed and accuracy on mobile devices.
    -   **Outcome:** A single, optimized model file (e.g., `fgo_detector.tflite`).

-   **Action 1.3: Create the `ml-recognizer` Gradle Module**
    -   **Purpose:** A self-contained module for all screen recognition tasks.
    -   **Key Components:**
        1.  The trained `.tflite` model file included in the module's assets.
        2.  A `GameStateParser` class that accepts a screen bitmap.
        3.  Integration of the TensorFlow Lite Interpreter to run the model.
        4.  Logic to process the model's raw output into a structured `GameState` data class, providing a clean abstraction of the on-screen information.

---

## Phase 2: The "Brain" - `ml-agent` Module

This phase focuses on creating the decision-making core of the ML Mode.

-   **Action 2.1: Formalize the Reinforcement Learning Environment**
    -   **Task:** Define the problem space for the RL agent.
    -   **State Space:** The `GameState` object produced by the `ml-recognizer` module.
    -   **Action Space:** A comprehensive and discrete set of all possible in-game actions (e.g., use specific skill on a specific target, attack with a card combination, use an NP).
    -   **Reward Function:** A carefully designed function to guide the agent towards winning efficiently (e.g., large positive reward for winning, small negative reward per turn, etc.).

-   **Action 2.2: Train the RL Agent (Offline)**
    -   **Task:** Train the decision-making model using a large, pre-existing dataset. This is computationally expensive and must be done on a dedicated machine.
    -   **Model Choice:** A **Deep Q-Network (DQN)** is a strong candidate for this task.
    -   **Training Process:**
        1.  Collect a dataset of gameplay trajectories (`state, action, reward, next_state`) from expert human players or existing FGA scripts.
        2.  Use a Python framework (PyTorch, TensorFlow) to train the DQN model.
        3.  Convert the final trained agent into the TensorFlow Lite (`.tflite`) format.

-   **Action 2.3: Create the `ml-agent` Gradle Module**
    -   **Purpose:** A self-contained module for all decision-making logic.
    -   **Key Components:**
        1.  The trained DQN `.tflite` model included in the module's assets.
        2.  A `DecisionAgent` class that accepts a `GameState` object.
        3.  Logic to run the state through the model and select the action with the highest predicted Q-value.

---

## Phase 3: Integration & Execution

This final phase connects the new ML modules to the FGA application and exposes the feature to the user.

-   **Action 3.1: Create a New "ML Battle Mode" in the UI**
    -   **Task:** Add a new script option in the FGA user interface, such as **"ML-Powered Battle."**
    -   **Implementation:** This will be a new `Script` class within the `scripts` module that orchestrates the entire ML workflow, distinct from existing scripts.

-   **Action 3.2: Implement the ML Orchestration Loop**
    -   **Task:** Develop the main control loop that drives the ML Mode.
    -   **Workflow:**
        1.  User starts the "ML-Powered Battle" script.
        2.  The script captures the screen using `libautomata`.
        3.  The screenshot is passed to the `ml-recognizer` to get the current `GameState`.
        4.  The `GameState` is passed to the `ml-agent` to get the optimal `Action`.
        5.  A new `ActionTranslator` component maps the abstract `Action` to a concrete sequence of screen coordinates and gestures.
        6.  The gesture sequence is executed by the existing `libautomata.Clicker`.
        7.  The loop repeats until the battle ends.

-   **Action 3.3: (Advanced) Implement On-Device Learning**
    -   **Task:** Add functionality for the agent to learn and adapt over time.
    -   **Mechanism:**
        1.  Locally store gameplay trajectories (`state, action, reward, next_state`) in a database.
        2.  Utilize the **TensorFlow Lite on-device training APIs** to create a background task that fine-tunes the `ml-agent` model using the collected data, allowing it to personalize its strategy based on the user's servants and CEs. 