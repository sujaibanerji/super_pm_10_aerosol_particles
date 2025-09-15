Atmospheric Data Processing and Analysis
========================================

This repository contains Python scripts for processing and analyzing atmospheric measurement datasets 
(scattering, absorption, and correction factors) between 2010 and 2022.

The project reproduces the workflow tested in Python 3.9 with specific library versions to ensure consistency.

GitHub Repository: https://github.com/sujaibanerji/super_pm_10_aerosol_particles

------------------------------------------------------------
Requirements
------------------------------------------------------------

- Python 3.9.x (must be installed)
- Virtual environment (recommended)

------------------------------------------------------------
Setup Instructions
------------------------------------------------------------

1. Clone the repository

   git clone https://github.com/sujaibanerji/super_pm_10_aerosol_particles.git
   cd super_pm_10_aerosol_particles

2. Install Python 3.9 (if you don’t have it)

   - Windows: Download the installer from https://www.python.org/downloads/release/python-3913/
     During installation, make sure to check the box "Add Python to PATH".

   - Linux (Ubuntu/Debian): 
       sudo apt update
       sudo apt install python3.9 python3.9-venv python3.9-dev

   - macOS: Use Homebrew
       brew install python@3.9

3. Create and activate a virtual environment (Python 3.9)

   On Windows:
       py -3.9 -m venv venv39
       venv39\Scripts\activate

   On Linux/macOS:
       python3.9 -m venv venv39
       source venv39/bin/activate

4. Install dependencies

   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

------------------------------------------------------------
Data
------------------------------------------------------------

The dataset required to run the analysis is available on Google Drive:

Download Data: https://drive.google.com/file/d/1xNmhbH4w_zYlcYOL8R5fWhp0YriqodfF/view?usp=sharing

After downloading, extract the files into the data/ directory of the project.  

Your folder structure should look like:

project-root/
│
├── data/
│   ├── ae33/
│   ├── Particle18/
│   ├── Particle19/
│   ├── ...
│   ├── smr_20180101.csv
│   └── smr_20220101.csv
│
├── abs/
│   ├── ae31_abs_bc_pm1_SMEARii_2010_2017.txt
│   └── ae31_abs_bc_pm10_SMEARii_2006_2017.txt
│
├── revised_manuscript_10_super.py
├── requirements.txt
└── README.txt

------------------------------------------------------------
Running the Script
------------------------------------------------------------

Once the data is in place and dependencies installed, run:

   python revised_manuscript_10_super.py

This will execute the analysis and produce correction factor plots (2010–2022).

------------------------------------------------------------
Dependencies
------------------------------------------------------------

Main libraries used:

- pandas==1.5.3  
- numpy==1.23.5  
- matplotlib==3.5.3  
- seaborn==0.12.2  
- scipy==1.9.3  
- statsmodels==0.13.5  
- scikit-learn==1.2.2  
- ruptures==1.1.7  
- hmmlearn==0.2.8  

See requirements.txt for the full list.

------------------------------------------------------------
Reproducibility Notes
------------------------------------------------------------

- The project is fixed to Python 3.9 for compatibility with pandas 1.5.3 and related libraries.  
- Using newer Python versions (3.12, 3.13) may cause incompatibilities.  
- All relative paths (./data/...) ensure cross-platform reproducibility.

------------------------------------------------------------
Troubleshooting
------------------------------------------------------------

If you face issues when installing dependencies (especially on Windows) where pip seems to use the wrong Python version 
(for example, Python 3.13 instead of Python 3.9), here is the solution that worked:

Although the virtual environment (venv39) was activated, pip was still pointing to Python 3.13 global installation.  
This happens because in Windows, sometimes the global pip mixes with the virtual environment pip.

How to ensure you use the correct pip:

Inside your virtual environment (venv39), run the following command explicitly:

   .\venv39\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

This guarantees that pip and the build tools are updated inside venv39 (Python 3.9), 
not in the global Python 3.13 installation.

After that, install your requirements again:

   .\venv39\Scripts\python.exe -m pip install -r requirements.txt
