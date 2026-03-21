"""
DeepTox — Flask Backend
Run from the /backend directory:  python app.py
Serves on http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import os

app = Flask(__name__)
CORS(app)   # allows the frontend (any origin) to call this API

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load UniTox Clinical Database ─────────────────────────────────────────
try:
    unitox_df = pd.read_csv(os.path.join(BASE_DIR, 'UniTox.csv'))
    unitox_df['Clean_Name'] = (
        unitox_df['Generic Name'].astype(str).str.lower().str.strip()
    )
    print("✅ UniTox clinical database loaded successfully.")
except Exception as e:
    unitox_df = None
    print(f"❌ UniTox load error: {e}")

# ─── Load Tox21 AI Model ────────────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(BASE_DIR, 'tox21_model.joblib'))
    print("✅ Tox21 AI model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Model load error: {e}")

# ─── RDKit Fingerprint Generator ────────────────────────────────────────────
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ─── Organ-readable label map ───────────────────────────────────────────────
organ_map = {
    'Nuclear Receptor-AhR':         'Liver Toxicity (AhR)',
    'Nuclear Receptor-PPAR-gamma':  'Metabolism / Liver',
    'Stress Response-p53':          'Tumor Risk (DNA Damage)',
    'Stress Response-Mitochondrial':'Heart / Brain Energy',
    'Nuclear Receptor-Estrogen':    'Fertility (Estrogen)',
    'Nuclear Receptor-Androgen':    'Fertility (Androgen)',
}

MODEL_TARGETS = [
    'Nuclear Receptor-Androgen',
    'Nuclear Receptor-Estrogen',
    'Nuclear Receptor-AhR',
    'Nuclear Receptor-Aromatase',
    'Nuclear Receptor-ER-LBD',
    'Nuclear Receptor-PPAR-gamma',
    'Stress Response-ARE',
    'Stress Response-ATAD5',
    'Stress Response-HSE',
    'Stress Response-MMP',
    'Stress Response-p53',
    'Stress Response-Mitochondrial',
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def rating_to_status(val):
    """Convert binary rating value to toxic / safe / unknown."""
    v = str(val).strip().lower()
    if v in ['1', 'yes', 'true', 'toxic', 'positive']:
        return 'toxic'
    elif v in ['0', 'no', 'false', 'safe', 'negative', 'nontoxic']:
        return 'safe'
    else:
        return 'unknown'


def get_clinical_report(drug_name: str) -> dict:
    """Search UniTox.csv and return structured organ toxicity data."""
    if unitox_df is None:
        return {'found': False, 'error': 'UniTox database not loaded.'}

    match = unitox_df[unitox_df['Clean_Name'] == drug_name.lower().strip()]

    if match.empty:
        return {'found': False}

    row = match.iloc[0]

    def safe_reason(col):
        val = str(row.get(col, 'No details available.'))
        return val

    organs = [
        {
            'name':      'Heart',
            'rating':    rating_to_status(row.get('Cardiotoxicity Binary Rating',   'N/A')),
            'reasoning': safe_reason('Cardiotoxicity Reasoning'),
        },
        {
            'name':      'Liver',
            'rating':    rating_to_status(row.get('LiverToxicity Binary Rating',    'N/A')),
            'reasoning': safe_reason('LiverToxicity Reasoning'),
        },
        {
            'name':      'Kidney',
            'rating':    rating_to_status(row.get('RenalToxicity Binary Rating',    'N/A')),
            'reasoning': safe_reason('RenalToxicity Reasoning'),
        },
        {
            'name':      'Lungs',
            'rating':    rating_to_status(row.get('PulmonaryToxicity Binary Rating','N/A')),
            'reasoning': safe_reason('PulmonaryToxicity Reasoning'),
        },
        {
            'name':      'Fertility',
            'rating':    rating_to_status(row.get('Infertility Binary Rating',      'N/A')),
            'reasoning': safe_reason('Infertility Reasoning'),
        },
    ]
    return {'found': True, 'organs': organs}


def predict_molecular_risk(smiles: str) -> dict:
    """Run Tox21 Random Forest model and return per-assay risk scores."""
    if model is None:
        return {'error': 'AI model not loaded.'}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES string — could not parse molecule.'}

        fp = mfpgen.GetFingerprintAsNumPy(mol).reshape(1, -1)
        probs = model.predict_proba(fp)

        results = []
        if isinstance(probs, list):
            for i, target_prob in enumerate(probs):
                if i >= len(MODEL_TARGETS):
                    break
                # Some tasks may have only 1 class (all-safe in training)
                risk_score = float(target_prob[0][1]) if target_prob.shape[1] > 1 else 0.0
                name     = MODEL_TARGETS[i]
                readable = organ_map.get(name, name)
                results.append({
                    'assay':    name,
                    'effect':   readable,
                    'risk_pct': round(risk_score * 100, 1),
                })

        results.sort(key=lambda x: x['risk_pct'], reverse=True)
        return {'predictions': results}

    except Exception as e:
        return {'error': str(e)}


def compute_overall_risk(clinical: dict, ai_data: dict) -> str:
    """Derive a single overall risk level from both layers."""
    pcts = [r['risk_pct'] for r in ai_data.get('predictions', [])]
    max_pct = max(pcts) if pcts else 0
    avg_pct = (sum(pcts) / len(pcts)) if pcts else 0

    if max_pct > 55 or avg_pct > 30:
        level = 'high'
    elif max_pct > 25 or avg_pct > 12:
        level = 'moderate'
    else:
        level = 'safe'

    # Boost level if clinical evidence is strong
    if clinical.get('found'):
        toxic_count = sum(1 for o in clinical['organs'] if o['rating'] == 'toxic')
        if toxic_count >= 3:
            level = 'high'
        elif toxic_count >= 1 and level == 'safe':
            level = 'moderate'

    return level


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':         'ok',
        'model_loaded':   model is not None,
        'unitox_loaded':  unitox_df is not None,
        'version':        '1.0.0',
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    compound   = data.get('compound', '').strip()
    input_type = data.get('input_type', 'name')   # 'name' | 'smiles'

    if not compound:
        return jsonify({'error': 'compound field is required.'}), 400

    # ── Resolve SMILES ──────────────────────────────────────────────────────
    drug_name = ''
    smiles    = ''

    if input_type == 'name':
        drug_name = compound
        try:
            results = pcp.get_compounds(compound, 'name')
            if not results:
                return jsonify({'error': f'"{compound}" not found in PubChem.'}), 404
            smiles = results[0].isomeric_smiles or results[0].smiles or ''
        except Exception as e:
            return jsonify({'error': f'PubChem lookup failed: {str(e)}'}), 500
    else:
        smiles    = compound
        drug_name = 'Unknown Chemical'

    if not smiles:
        return jsonify({'error': 'Could not resolve a SMILES for this compound.'}), 400

    # ── Run Both Layers ─────────────────────────────────────────────────────
    clinical = get_clinical_report(drug_name)
    ai_data  = predict_molecular_risk(smiles)
    overall  = compute_overall_risk(clinical, ai_data)

    return jsonify({
        'name':         drug_name,
        'smiles':       smiles,
        'clinical':     clinical,
        'ai_risks':     ai_data.get('predictions', []),
        'overall_risk': overall,
        'ai_error':     ai_data.get('error'),
    })


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🧪 DeepTox Flask Server Starting...")
    print("📡 API available at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
