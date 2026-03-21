# 🧬 DeepTox — Hybrid AI Toxicity Prediction System

**A two-layer safety architecture that bridges verified FDA clinical records with NIH machine-learning predictions — giving you answers no single model can.**

🌐 **[View Live Demo](https://adityakumar008.github.io/DeepTox/)**

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [About the Project](#about-the-project)
- [Demo Video](#demo-video)
- [Live Demo](#live-demo)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Datasets Used](#datasets-used)
- [ML Model and Training](#ml-model-and-training)
- [Folder Structure](#folder-structure)
- [Run Locally](#run-locally)
- [Features](#features)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## 🎯 Problem Statement

Toxicity prediction is a critical step in drug discovery and chemical safety evaluation. Currently, two types of systems exist — but both have serious limitations:

- **Clinical databases** (like FDA drug labels) are authoritative and human-verified, but they only cover drugs that have already been tested on humans. They are completely blind to novel or untested compounds.
- **AI/ML models** (trained on lab data) can assess any chemical structure, but their predictions are probabilistic and based on in-vitro cell experiments — which don't always reflect real human responses.

**The gap:** No existing lightweight tool combines both sources simultaneously and presents their findings side by side in a transparent, interpretable way.

**The solution — DeepTox:** A hybrid system that runs both a clinical database scan and an AI molecular risk prediction in parallel, clearly labeling the source and confidence of each finding, so the user always knows what is verified fact vs. what is a model estimate.

---

## 🔬 About the Project

**DeepTox** is a hybrid AI toxicity prediction system built as part of academic research at IIT Jodhpur.

**DeepTox runs two independent layers simultaneously:**

| Layer | Source | Type | Strength |
|-------|--------|------|----------|
| 🏥 **Layer 1 — Clinical Doctor** | UniTox (FDA drug labels) | In-vivo · Human | 100% authoritative for known drugs |
| 🧠 **Layer 2 — AI Chemist** | Tox21 (NIH / EPA) | In-vitro · ML | Works on any chemical, even novel ones |

The result is a system that is more complete than either layer alone — and honest about what each layer does and doesn't know.

---

## 🎬 Demo Video

> ▶️ **[Watch Full Demo on Google Drive](https://drive.google.com/file/d/1FIYSW5EjaiTjz7ji75sZZmChX9K7paqe/view?usp=drivesdk)**

The video demonstrates the complete working system — Flask backend starting in terminal, live predictions from the ML model, 2D molecular structure rendering, clinical records with doctor reasoning, AI risk bars, light/dark mode toggle, and PDF report download.

---

## 🌐 Live Demo

> 🔗 **[Click here to view DeepTox Live](https://adityakumar008.github.io/DeepTox/)**

The live demo runs in **Demo Mode** (mock data) since the Flask backend cannot run on GitHub Pages (static hosting only).

To get **real predictions** from the actual ML model, follow the [Run Locally](#run-locally) instructions below.

---

## ⚙️ How It Works

Step-by-step flow when a user searches for a compound:

```
User Input (Drug Name or SMILES)
        │
        ▼
[PubChemPy] ── If name given, fetch SMILES from PubChem API
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[Layer 1 — Pandas]                    [Layer 2 — RDKit + ML]
Search UniTox.csv                     Generate Morgan Fingerprint
Return organ toxicity ratings         Feed to Random Forest model
+ Doctor's clinical reasoning         Get risk % per assay target
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              [Flask API — /analyze]
              Compile both results into JSON
                       │
                       ▼
              [Frontend — index.html]
              Render 2D molecule structure (RDKit.js)
              Clinical table + AI risk bars
              Overall risk verdict + PDF report
```

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **ML Engine** | scikit-learn (Random Forest) | Toxicity prediction |
| **Cheminformatics** | RDKit | SMILES parsing + Morgan fingerprints |
| **Data Management** | Pandas | UniTox clinical database lookup |
| **SMILES Resolver** | PubChemPy | Drug name → SMILES via PubChem API |
| **Backend** | Flask + Flask-CORS | REST API connecting ML to frontend |
| **Frontend** | HTML / CSS / JavaScript | Dark-themed SPA dashboard |
| **Molecule Renderer** | RDKit.js | 2D chemical structure visualization in browser |
| **PDF Reports** | jsPDF | Client-side clean white report generation |

---

## 📊 Datasets Used

### 1. UniTox — Clinical Database
- **Source:** FDA Drug Labels
- **Type:** In-vivo · Real-world human data
- **Coverage:** Heart, Liver, Kidney, Lungs, Fertility
- **Annotations:** Binary toxicity rating + physician-documented reasoning
- **File:** `UniTox.csv`

### 2. Tox21 — ML Training Data
- **Source:** NIH / EPA / NCATS (Tox21 Initiative)
- **Type:** In-vitro · Experimental lab data
- **Compounds:** 12,707 chemicals
- **Assay Targets:** 12 (Nuclear Receptors + Stress Response pathways)
- **File:** `Tox21.csv`

#### Tox21 Assay Targets:
```
Nuclear Receptors (NR):          Stress Response (SR):
├── NR-AR (Androgen Receptor)    ├── SR-ARE
├── NR-AhR                       ├── SR-ATAD5
├── NR-Aromatase                 ├── SR-HSE
├── NR-ER                        ├── SR-MMP
├── NR-ER-LBD                    └── SR-p53 / SR-Mitochondrial
└── NR-PPAR-gamma
```

---

## 🤖 ML Model and Training

### Model: Random Forest Classifier
- **Algorithm:** Multi-output Random Forest Classifier
- **Input:** 2048-bit Morgan Fingerprint (radius=2) generated by RDKit
- **Output:** Toxicity probability (0–100%) per assay target
- **Training Data:** Tox21 dataset (12,707 compounds × 12 targets)

### Model File Not Included
The trained model file `tox21_model.joblib` is **not included** in this repository due to its file size (116 MB), which exceeds GitHub's limit.

### How to Train the Model Yourself

1. Make sure `Tox21.csv` is in your project folder
2. Run the training script:

```python
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load dataset
df = pd.read_csv('Tox21.csv')

# Generate fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return mfpgen.GetFingerprintAsNumPy(mol)
    return None

df['fp'] = df['smiles'].apply(smiles_to_fp)
df = df.dropna(subset=['fp'])

X = np.stack(df['fp'].values)
target_cols = [c for c in df.columns if c != 'smiles']
y = df[target_cols].fillna(0).astype(int)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(rf)
model.fit(X, y)

# Save model
joblib.dump(model, 'tox21_model.joblib')
print("✅ Model saved!")
```

3. Place the generated `tox21_model.joblib` alongside `app.py` and run the server

---

## 📁 Folder Structure

```
DeepTox/
│
├── index.html              # Frontend — full SPA website
├── app.py                  # Backend — Flask REST API
│
├── UniTox.csv              # Clinical database (FDA drug labels)
├── Tox21.csv               # ML training data (NIH Tox21)
├── tox21_model.joblib      # NOT INCLUDED — train locally (116MB)
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/AdityaKumar008/DeepTox.git
cd DeepTox
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model (first time only)
Follow the [training instructions above](#how-to-train-the-model-yourself) to generate `tox21_model.joblib`.

### Step 4 — Start the Flask backend
```bash
python app.py
```
Server starts at `http://127.0.0.1:5000`

### Step 5 — Open the frontend
Open `index.html` in your browser. The status badge in the bottom-right corner will switch from **DEMO MODE** to **FLASK LIVE** automatically. ✅

---

## ✨ Features

### Core Analysis
- 🔍 **Dual-source analysis** — Clinical FDA records + AI predictions side by side
- 💊 **Drug name lookup** — Type any common name, PubChem resolves the structure automatically
- 🧪 **SMILES support** — Directly input chemical structures for novel compounds
- 📊 **Visual risk bars** — Color-coded probability bars for all 12 assay targets

### Molecule Visualizer
- ⚛️ **2D chemical structure rendering** — Live molecular diagram rendered in the browser using RDKit.js
- 🔢 **Molecular properties** — Formula, molecular weight, heavy atom count, ring count, H-bond donors and acceptors
- 📋 **Copy SMILES** — One-click copy of the resolved SMILES string

### Usability
- 🕓 **Search history** — Last 4 searches saved with risk level and timestamp, click any to re-run
- 🌙 **Light / Dark mode** — Toggle with memory across sessions
- ⚡ **Smart fallback** — Demo mode when Flask isn't running, great for presentations
- 📱 **Responsive design** — Works on desktop and mobile

### Reports
- ⬇️ **PDF report download** — Clean white A4 report with drug info, clinical table, AI risk bars, summary, and disclaimer — ready to print or share

---

## ⚠️ Disclaimer

DeepTox is an **academic research prototype** developed for educational purposes only. The predictions generated — whether from the clinical database or the AI model — are **not validated for clinical use** and must not be used to make any medical, pharmaceutical, or safety decisions. Always consult qualified medical professionals.

---

## 👨‍💻 Author

**Aditya Kumar**  
LinkedIn: https://www.linkedin.com/in/aditya-kumar-b7920b328

> *Project developed as part of academic coursework at IIT Jodhpur*

---

<div align="center">
  <sub>Built with 🧬 by Aditya Kumar · © 2025 DeepTox · Research Prototype</sub>
</div>
