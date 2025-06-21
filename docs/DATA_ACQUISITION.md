# FGO-Bot-ML: Data Acquisition Strategy

This document outlines the methods and procedures for acquiring the necessary data to train the machine learning models for the FGO-Bot-ML enhancement. Data will be collected for two primary purposes: training the Computer Vision model for entity recognition and training the Reinforcement Learning model for decision-making.

## 1. Vision Model Data (Servant & Craft Essence Images)

The goal is to build a comprehensive image dataset of every Servant and Craft Essence in the game. This dataset needs to be diverse to ensure the model is robust to in-game variations.

### Methods of Acquisition

1.  **Automated In-Game Screenshotting (Primary Method):**
    *   **Concept:** Develop a dedicated script using the existing `libautomata` framework to systematically navigate through the game and capture screenshots of all Servants and CEs.
    *   **Execution Plan:**
        *   The script will navigate to `My Room -> Spirit Origin List`.
        *   It will iterate through every Servant and CE entry.
        *   For each entry, it will open the details page and capture images of all ascension/costume art.
        *   The script will also navigate to the Party Setup screen to capture smaller "portrait" style images.
        *   Captured images will be saved locally with a systematic naming convention (e.g., `servant_id_ascension_1.png`).

2.  **Web Scraping from Community Databases (Supplementary Method):**
    *   **Concept:** Augment our dataset by downloading images from reputable FGO community resources.
    *   **Target Sources:**
        *   [Atlas Academy DB](https://atlasacademy.io/db/)
        *   [FGO GamePress Wiki](https://gamepress.gg/grandorder/)
        *   [Fate/Grand Order Wiki](https://fategrandorder.fandom.com/wiki/Fate/Grand_Order_Wikia)
    *   **Execution Plan:**
        *   Write a Python script using libraries like `BeautifulSoup` and `Requests`.
        *   The script will scrape the websites for high-quality servant and CE card art.
        *   This method is particularly useful for obtaining clean, high-resolution base images.

3.  **Manual Data Collection:**
    *   **Concept:** Manually take screenshots of specific in-game situations that are difficult to automate.
    *   **Use Cases:**
        *   Capturing images of new Servants/CEs immediately after release.
        *   Getting images of specific battle UI elements, buffs, or enemy sprites that may not be available in databases.

4.  **Data Augmentation:**
    *   **Concept:** Programmatically increase the size and variance of our dataset.
    *   **Techniques:**
        *   **Geometric Transformations:** Random rotation, scaling, and cropping.
        *   **Color and Lighting Adjustments:** Varying brightness, contrast, and saturation.
        *   **Noise Injection:** Adding random noise to simulate different screen conditions.
    *   **Execution:** A script will be created to apply these augmentations to the collected images, creating a larger and more robust training set.

## 2. Reinforcement Learning Model Data (Gameplay Sequences)

The goal is to collect a large set of state-action-reward sequences from actual gameplay. This data will be used to teach the RL agent what actions to take in a given situation.

### Methods of Acquisition

1.  **Recording Expert Bot Gameplay (Primary Method):**
    *   **Concept:** Leverage the existing, highly-effective FGA bot. We will configure it with optimized scripts for various quests (especially farming nodes) and record its "expert" behavior.
    *   **Execution Plan:**
        *   Enhance the bot's core loop with a recording module.
        *   When recording is active, for each step in the script, the module will save:
            *   **State:** The current screen capture.
            *   **Action:** The command executed by the script (e.g., `useSkill(1)`, `selectCard(3)`).
            -   **Metadata:** Parsed game state if available (e.g., turn number from OCR).
        *   This will generate large quantities of consistent, high-quality data for common gameplay scenarios.

2.  **Recording Human Gameplay (Supplementary Method):**
    *   **Concept:** Create a tool that allows human players to record their own gameplay sessions. This is invaluable for capturing complex strategies and decision-making that is difficult to script.
    *   **Execution Plan:**
        *   Develop a "recording mode" in the app.
        *   This mode will run a background service that captures the screen and records all user taps and swipes.
        *   The raw input data will be correlated with screen captures to create state-action pairs.
        *   A simple UI will allow users to start and stop recording and upload the data.

---

This dual approach of automated and manual, bot and human data collection will provide a rich and diverse dataset, forming a strong foundation for our ML models. 