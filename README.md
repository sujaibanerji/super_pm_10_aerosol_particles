# Atmospheric Data Processing and Analysis

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3913/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains Python scripts for processing and analyzing atmospheric measurement datasets 
(scattering, absorption, and correction factors) between **2010 and 2022**.

The workflow was tested in **Python 3.9** with pinned dependencies to ensure reproducibility.

📂 GitHub Repository: [super_pm_10_aerosol_particles](https://github.com/sujaibanerji/super_pm_10_aerosol_particles)

---

## 🚀 Requirements

- Python **3.9.x** (must be installed)
- Virtual environment (recommended)

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/sujaibanerji/super_pm_10_aerosol_particles.git
cd super_pm_10_aerosol_particles
```

2. **Install Python 3.9 (if not installed)**

- **Windows**: [Download Python 3.9.13](https://www.python.org/downloads/release/python-3913/)  
  Be sure to check **"Add Python to PATH"** during installation.

- **Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

- **macOS** (using Homebrew):
```bash
brew install python@3.9
```

3. **Create and activate a virtual environment**

- Windows:
```powershell
py -3.9 -m venv venv39
venv39\Scripts\activate
```

- Linux/macOS:
```bash
python3.9 -m venv venv39
source venv39/bin/activate
```

4. **Install dependencies**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 📊 Data

The dataset required for the analysis is available on Google Drive:

👉 [Download Data](https://drive.google.com/file/d/1xNmhbH4w_zYlcYOL8R5fWhp0YriqodfF/view?usp=sharing)

After downloading, extract the files into the `data/` directory.

### Folder structure

```text
project-root/
│
├── data/
│   ├── ae33/
│   │   ├── smr_20180101.csv
│   │   ├── smr_20190101.csv
│   │   ├── smr_20200101.csv
│   │   ├── smr_20210101.csv
│   │   └── smr_20220101.csv
│   ├── Particle18/
│   ├── Particle19/
│   └── ...
│
├── abs/
│   ├── ae31_abs_bc_pm1_SMEARii_2010_2017.txt
│   └── ae31_abs_bc_pm10_SMEARii_2006_2017.txt
│
├── revised_manuscript_10_super.py
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Script

After placing the data and installing dependencies, run:

```bash
python revised_manuscript_10_super.py
```

This will execute the full analysis and generate correction factor plots (**2010–2022**).

---

## 📦 Dependencies

Main libraries:

- pandas==1.5.3  
- numpy==1.23.5  
- matplotlib==3.5.3  
- seaborn==0.12.2  
- scipy==1.9.3  
- statsmodels==0.13.5  
- scikit-learn==1.2.2  
- ruptures==1.1.7  
- hmmlearn==0.2.8  

See `requirements.txt` for the full list.

---

## 🔁 Reproducibility Notes

- This project is pinned to **Python 3.9** for compatibility.  
- Using Python 3.12 or 3.13 may cause incompatibility issues.  
- All file paths are **relative** (`./data/...`) to ensure cross-platform reproducibility.  

---

## 🛠 Troubleshooting

If you face issues with `pip` using the wrong Python version (e.g., pointing to **Python 3.13** instead of **3.9**), use this workaround:

```powershell
.\venv39\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\venv39\Scripts\python.exe -m pip install -r requirements.txt
```

This ensures that pip and dependencies are properly installed in **Python 3.9 (venv39)** and not mixed with global Python versions.
