# FGO-Bot-ML: Machine Learning Data Collection

This directory contains the machine learning data collection scripts and datasets for the FGO-Bot-ML project.

## 📁 Directory Structure

```
ml/
├── dataset/                    # Downloaded image datasets
│   ├── servants/              # Atlas Academy servant images (~1GB)
│   └── ces/                   # Atlas Academy craft essence images (~1.3GB)
├── download_servant_images.py # Servant data collection script
├── download_ce_images.py      # Craft essence data collection script
├── requirements.txt           # Python dependencies
├── .venv/                     # Python virtual environment
└── README.md                  # This file
```

## 🚀 Quick Start

### Setup Environment

```bash
# Navigate to ml directory
cd ml

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Download all servant images and metadata
python download_servant_images.py

# Download all craft essence images and metadata
python download_ce_images.py
```

## 📊 Current Dataset Status

| Dataset | Size | Images | Status |
|---------|------|--------|--------|
| Servants | ~1.0GB | ~5,000+ | ✅ Complete |
| Craft Essences | ~1.3GB | ~6,000+ | ✅ Complete |
| **Total** | **~2.3GB** | **~11,265** | **✅ Ready for ML** |

## 🔧 Available Scripts

### `download_servant_images.py`
Downloads comprehensive servant data from Atlas Academy API.

**Features:**
- All servant ascension artwork
- Costume variants and special artwork
- Complete metadata with stats, skills, and attributes
- Structured directory organization
- Comprehensive error handling and logging

**Output Structure:**
```
dataset/servants/{servant_id}/
├── metadata.json              # Complete servant data
├── ascension_stage_1_{id}.png
├── ascension_stage_2_{id}.png
├── costume_variant_{id}.png
└── servant_icon_{id}.png
```

### `download_ce_images.py`
Downloads craft essence data from Atlas Academy API.

**Features:**
- All CE card artwork in multiple sizes
- Icon variants and face images
- Complete metadata with effects and stats
- Structured directory organization
- Comprehensive error handling and logging

**Output Structure:**
```
dataset/ces/{ce_id}/
├── metadata.json              # Complete CE data
├── card_art_{id}.png
├── face_image_{id}.png
└── icon_{id}.png
```

## 📝 Data Sources

**Primary Source:** [Atlas Academy API](https://api.atlasacademy.io/)
- Reliable, well-maintained public API
- Complete coverage of FGO game assets
- High-resolution official artwork
- Comprehensive metadata

## 🎯 Next Steps

The dataset is now ready for:
1. **Computer Vision Model Training** - Image classification and recognition
2. **Data Preprocessing** - Image normalization and augmentation
3. **Feature Extraction** - Preparing data for ML pipelines
4. **Model Development** - Training vision models for FGO entity recognition

## 🔄 Maintenance

To update the datasets with new content:
```bash
# Re-run the scripts to fetch latest data
python download_servant_images.py
python download_ce_images.py
```

The scripts automatically handle:
- Incremental updates (only downloads new/missing data)
- Error recovery and retry mechanisms
- Rate limiting to respect API usage policies
- Detailed logging for monitoring progress

---

For more information about the overall data acquisition strategy, see [`../docs/DATA_ACQUISITION.md`](../docs/DATA_ACQUISITION.md). 