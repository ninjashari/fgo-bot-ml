# FGO-Bot-ML: Machine Learning Feature Requirements

This document specifies the detailed requirements for the machine learning-enhanced battle module. It expands upon the action plan by defining the specific components, data structures, and logic required for the system to function.

## 1. System Overview

The ML-driven battle system will consist of two primary intelligent components:

1.  **Vision System:** A computer vision model responsible for identifying all relevant game entities on the screen.
2.  **Decision Engine:** A reinforcement learning (RL) agent that selects the optimal in-game action based on the state provided by the Vision System.

These components will be orchestrated by a central `MLBattleManager` which interfaces with the main application and the existing `libautomata` for command execution.

## 2. Vision System Requirements

The Vision System's purpose is to convert raw pixel data from a screenshot into a structured representation of the game state.

### 2.1. Identified Entities

The system must be able to identify and locate the following:

-   **Player Servants (1-3):** Including which Servant is in which slot.
-   **Enemy Servants/Mobs (1-3+):** Including their position.
-   **Command Cards (5):** Type (Buster, Arts, Quick), associated Servant, and any status effects (e.g., crit stars).
-   **Servant Skills (9):** Usable, in-cooldown, or unavailable.
-   **Noble Phantasm (NP) Buttons (3):** Ready or not ready.
-   **Attack/Back/Menu Buttons:** To understand the current UI state.
-   **Active Buffs/Debuffs:** On both player and enemy servants (stretch goal, high complexity).

### 2.2. Output Data Structure

The Vision System's output for a given screenshot will be a JSON-like object or a set of Kotlin data classes representing the complete, structured battle state.

**Example Structure:**

```kotlin
data class BattleState(
    val playerServants: List<ServantState>,
    val enemyServants: List<EnemyState>,
    val commandCards: List<CommandCard>,
    val skills: List<SkillState>,
    val noblePhantasms: List<NoblePhantasmState>,
    val currentTurn: Int,
    val currentWave: Int
    // ... and other relevant data
)
```

## 3. Decision Engine Requirements

The Decision Engine is the core of the ML feature, responsible for strategy and action selection.

### 3.1. State Space (Input to RL Model)

The state provided to the RL model must be a fixed-size numerical vector (a tensor). This vector will be a flattened and normalized representation of the `BattleState` object. It must encode all information necessary for making a strategic decision, including:

-   HP (as a percentage) for all player and enemy servants.
-   NP gauge (as a percentage) for all player servants.
-   Skill availability (binary: ready/not ready).
-   NP availability (binary: ready/not ready).
-   Command card types and servant affinity.
-   Current wave and turn number.

### 3.2. Action Space (Output of RL Model)

The model will output a single value indicating which action to take from a predefined, discrete set of all possible actions.

**Action Set Definition:**

-   `USE_SKILL_1` through `USE_SKILL_9`
-   `SELECT_NP_1` through `SELECT_NP_3`
-   `SELECT_CARD_1` through `SELECT_CARD_5`
-   `PRESS_ATTACK_BUTTON`
-   `TARGET_ENEMY_1` through `TARGET_ENEMY_3`
-   ... and any other necessary discrete actions.

The `MLBattleManager` will be responsible for interpreting this action and filtering out invalid moves (e.g., trying to use a skill on cooldown).

### 3.3. Reward Function

The reward function is critical for training an effective agent. The function will be a weighted sum of several factors, calculated after each turn or battle.

-   **Primary Reward (Quest Completion):**
    -   Large positive reward for winning the battle.
    -   Large negative reward for losing the battle.

-   **Intermediate Rewards (Turn-by-Turn):**
    -   **Positive:**
        -   Damage dealt to enemies.
        -   Defeating an enemy.
        -   Clearing a wave.
        -   Efficient turn count (e.g., bonus for 3-turning a farming node).
    -   **Negative:**
        -   HP lost by player Servants.
        -   A player Servant being defeated.
        -   Wasted actions (e.g., over-charging an NP gauge).

The specific weights for each component will be hyperparameters that require tuning during the training process.

## 4. Integration and Control Flow

1.  The user selects "ML Battle Mode" from the UI.
2.  The `MLBattleManager` is initialized.
3.  The battle begins.
4.  **Loop:**
    a. The `MLBattleManager` captures the screen.
    b. It passes the image to the `VisionSystem` to get the current `BattleState`.
    c. The `BattleState` is converted into a numerical tensor.
    d. The tensor is fed into the `DecisionEngine`.
    e. The `DecisionEngine` outputs a chosen action.
    f. The `MLBattleManager` translates the action into a command for `libautomata` (e.g., `click(Location)`).
    g. `libautomata` executes the command.
    h. A short delay is introduced to allow the game animation to complete.
    i. The loop repeats until the battle ends.
5.  Upon battle completion, the final reward is calculated and (for online learning) used to update the model. 