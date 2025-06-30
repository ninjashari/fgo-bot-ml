# FGO-Bot-ML: Data Acquisition Strategy

This document outlines the methods and procedures for acquiring the necessary data to train the machine learning models for the FGO-Bot-ML enhancement. Data will be collected for two primary purposes: training the Computer Vision model for entity recognition and training the Reinforcement Learning model for decision-making.

## 1. Vision Model Data (Servant & Craft Essence Images)

The goal is to build a comprehensive image dataset of every Servant and Craft Essence in the game. This dataset needs to be diverse to ensure the model is robust to in-game variations.

### Current Implementation: Atlas Academy API

**Primary Data Source:** [Atlas Academy DB API](https://api.atlasacademy.io/)

Our implementation uses the Atlas Academy's comprehensive public JSON API as the sole data source for image collection. This approach provides:

- **Reliability:** Stable, well-maintained API with consistent data structure
- **Completeness:** Comprehensive coverage of all Servants and Craft Essences
- **Quality:** High-resolution official game assets
- **Maintainability:** No web scraping complexity or SSL certificate issues

### Implementation Details

**Servants Dataset (`dataset/servants/`)**
- **Source:** Atlas Academy API endpoints for servant data
- **Coverage:** All servants with complete ascension and costume artwork
- **Structure:** Individual directories per servant containing:
  - `metadata.json`: Complete servant data from API
  - Multiple image files: ascension art, costume variants, icons
- **Current Status:** ~1GB of data successfully collected

**Craft Essences Dataset (`dataset/ces/`)**
- **Source:** Atlas Academy API endpoints for CE data  
- **Coverage:** All craft essences with official artwork
- **Structure:** Individual directories per CE containing:
  - `metadata.json`: Complete CE data from API
  - Image files: card art, face images, icons in various sizes
- **Current Status:** ~1.3GB of data successfully collected

### Scripts and Tools

1. **`download_servant_images.py`** - Downloads servant data from Atlas Academy
2. **`download_ce_images.py`** - Downloads craft essence data from Atlas Academy

Both scripts include:
- Comprehensive error handling and logging
- Respectful rate limiting to avoid API abuse
- Structured metadata storage for ML pipeline integration
- Progress tracking with detailed status reporting

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

## Current Dataset Summary

**Total Images Collected:** ~11,265 images  
**Total Dataset Size:** ~2.3GB  
**Coverage:** Complete Atlas Academy servant and CE collections  
**Quality:** High-resolution official game assets  
**Structure:** ML-ready with comprehensive metadata  

This streamlined approach using Atlas Academy as our sole data source provides a robust foundation for computer vision model training while maintaining code simplicity and reliability. 