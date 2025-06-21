# FGO-Bot-ML: Machine Learning Enhancement Action Plan

This document outlines the comprehensive action plan for integrating a machine learning-driven decision-making engine into the FGO-Bot application. The goal is to create a system that is aware of in-game entities (Servants, Craft Essences), can make strategic decisions autonomously, and learns from its gameplay to improve performance over time.

This feature will be developed as a modular extension, ensuring no disruption to the existing codebase while leveraging its automation capabilities.

## Guiding Principles

- **Modularity:** The ML system will be developed in a separate, self-contained module (`ml`) to ensure clean separation of concerns.
- **Extensibility:** The design will be flexible to accommodate new Servants, CEs, and game mechanics with minimal changes.
- **Performance:** On-device inference will be prioritized, using optimized models (e.g., TensorFlow Lite) to ensure minimal impact on device performance and battery life.
- **User Transparency:** The UI will provide feedback on the ML agent's decisions and state, building user trust and aiding in diagnostics.

---

## Phase 1: Foundation, Project Setup, and Data Acquisition

The primary goal of this phase is to lay the groundwork for the ML feature by establishing the project structure and initiating the critical process of data collection.

1.  **Create Documentation:**
    -   `docs/ML_ACTION_PLAN.md`: This document.
    -   `docs/ML_FEATURE_REQUIREMENTS.md`: A detailed breakdown of the feature requirements, including state representation, action space, and reward function definitions.

2.  **Project Structure Setup:**
    -   Create a new Gradle module named `ml`. This module will house all ML-related code, including models, inference logic, and data processing utilities.

3.  **Data Acquisition - Vision Model:**
    -   Collect a comprehensive image dataset of all Servants and Craft Essences in the game.
    -   This dataset must include various representations:
        -   All ascension and costume variants.
        -   Card art, battle sprites, and support selection portraits.
        -   Images under different in-game lighting and background conditions.
    -   Utilize community resources (e.g., wikis) and automated screen capture scripts to build this dataset.

4.  **Data Acquisition - Reinforcement Learning Model:**
    -   Gather a large dataset of expert gameplay.
    -   This involves recording state-action-reward sequences from skilled human players or the existing, highly-configured bot.
    -   The `state` will include screen captures and parsed game data (HP, NP gauges, etc.).
    -   The `action` will be the recorded user input (skill activations, card selections).
    -   The `reward` will be calculated based on the outcome (damage dealt, enemies defeated, quest completion).

---

## Phase 2: Servant and Craft Essence Recognition (Computer Vision)

This phase focuses on developing the model that allows the application to "see" and understand the in-game environment.

1.  **Model Selection and Training:**
    -   Select a lightweight, efficient CNN architecture suitable for mobile deployment (e.g., MobileNetV3, EfficientNet-Lite).
    -   Train a multi-label image classification model on the dataset collected in Phase 1.
    -   Fine-tune pre-trained weights to accelerate training and improve accuracy.

2.  **Model Conversion and Integration:**
    -   Convert the trained model to the TensorFlow Lite (`.tflite`) format for on-device inference.
    -   Integrate the TFLite model and the TensorFlow Lite interpreter into the `ml` module.

3.  **Develop Vision API:**
    -   Create a high-level API within the `ml` module (e.g., `VisionManager`).
    -   This API will expose functions that take a game screenshot as input and return a structured list of identified Servants, CEs, and their locations on the screen.

---

## Phase 3: Intelligent Decision-Making (Reinforcement Learning)

This is the core phase where the "brain" of the bot is developed.

1.  **Formalize the RL Problem:**
    -   **State Space:** Define a precise, numerical representation of the game state based on data from the Vision API and screen scraping (HP, NP gauges, available skills, card types, wave number, etc.).
    -   **Action Space:** Define all possible actions the agent can take (e.g., use skill 1-9, attack with NP 1-3, select command cards 1-5).
    -   **Reward Function:** Engineer a sophisticated reward function that encourages strategic gameplay. This will include rewards for winning, penalties for losing, and intermediate rewards for clearing waves, defeating powerful enemies, and finishing turns efficiently.

2.  **RL Model Development:**
    -   Select and implement a suitable RL algorithm. A Deep Q-Network (DQN) or a similar value-based method is a strong starting point.
    -   Train the RL agent offline using the expert gameplay data collected in Phase 1. This pre-trains the model with effective strategies.

3.  **Model Conversion and Integration:**
    -   Convert the trained policy network (the "brain" of the RL agent) into the TensorFlow Lite format.
    -   Integrate the model into the `ml` module.

4.  **Develop Decision API:**
    -   Create an API within the `ml` module (e.g., `DecisionEngine`).
    -   This API will take the current game state as input and output the optimal action to be taken, as determined by the RL model.

---

## Phase 4: Integration and User Interface

This phase connects the new ML module to the existing application and exposes the new functionality to the user.

1.  **Bridge `app` and `ml` Modules:**
    -   In the `app` module, add the `ml` module as a dependency.
    -   Create a new "ML Battle Mode" option in the battle configuration screen.

2.  **Implement the ML Gameplay Loop:**
    -   When ML mode is active, the main script loop will:
        1.  Capture the current screen.
        2.  Call the `VisionManager` to get the current game state.
        3.  Call the `DecisionEngine` with the state to get the next action.
        4.  Translate the action into commands for the existing `libautomata` engine (e.g., `Clicker`, `Skill-use functions`).
        5.  Execute the command.
        6.  Repeat.

3.  **UI for ML Mode:**
    -   Add UI elements to provide feedback to the user, such as an overlay that highlights the agent's intended action or displays its confidence score.
    -   Create a new screen to show ML model statistics and performance.

---

## Phase 5: Online Learning, Testing, and Refinement

The final phase focuses on testing the integrated system and enabling it to learn and improve continuously.

1.  **End-to-End Testing:**
    -   Rigorously test the ML mode across a wide variety of nodes, team compositions, and in-game events.
    -   Identify and fix bugs in the state representation, decision logic, and integration points.

2.  **Implement Online Learning (Optional but Recommended):**
    -   Develop a mechanism to allow the RL model to continue learning during live gameplay.
    -   The agent will update its policy based on the outcomes of its own decisions, gradually refining its strategy.
    -   This requires careful implementation to ensure the agent doesn't learn detrimental behaviors.

3.  **Performance Monitoring and Retraining:**
    -   Collect gameplay data generated by the ML agent.
    -   Periodically use this new data to retrain and improve both the vision and RL models, ensuring the bot adapts to new game content and refines its strategies over time. 