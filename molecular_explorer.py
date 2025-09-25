# --- Core Imports ---
import os
import re
import sys
import base64
import datetime
import time
from io import BytesIO
from typing import Dict, Optional, List, Union
from collections import Counter

# Streamlit & UI
import streamlit as st
from streamlit.components.v1 import html as st_html

# Scientific Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Chemistry Libraries
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from pymatgen.core.composition import Composition 
from ase import Atoms 
from ase.calculators.emt import EMT 
from ase.optimize import BFGS 
import pubchempy as pcp
import py3Dmol
import speech_recognition as sr
import pyttsx3
import plotly.graph_objects as go
import plotly.express as px
import requests
# Local modules
sys.path.append(".")
import sascorer

# Load secret API key
api_key = st.secrets["OPENROUTER_API_KEY"]

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []


# Application configuration
st.set_page_config(
    page_title="Molecular Explorer Pro | by Aqsa Ijaz",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)
# 🎉 Play welcome voice only once
# Initialize voice engine once
engine = pyttsx3.init()

def speak(text):
    try:
        if not engine._inLoop:
            engine.say(text)
            engine.runAndWait()
    except RuntimeError:
        pass  # Prevent crash if another thread is already speaking
if 'has_welcomed' not in st.session_state:
    st.session_state.has_welcomed = True
    speak("Welcome! We'll explore chemistry together!")


# Unique chemical database
CHEMICAL_DATABASE = {
    "Methane (CH4)": "CH4", "Ethane (C2H6)": "C2H6", 
    "Propane (C3H8)": "C3H8", "Butane (C4H10)": "C4H10",
    "Methanol (CH3OH)": "CH3OH", "Ethanol (C2H5OH)": "C2H5OH",
    "Formic Acid (HCOOH)": "HCOOH", "Acetic Acid (CH3COOH)": "CH3COOH",
    "Water (H2O)": "H2O", "Ammonia (NH3)": "NH3", 
    "Carbon Dioxide (CO2)": "CO2", "Hydrogen (H2)": "H2",
    "Sodium Hydroxide (NaOH)": "NaOH", "Potassium Hydroxide (KOH)": "KOH",
    "Chlorine (Cl2)": "Cl2", "Fluorine (F2)": "F2",
    "Polyethylene ((C2H4)n)": "(C2H4)n", "Polystyrene ((C8H8)n)": "(C8H8)n",
    "Caffeine (C8H10N4O2)": "C8H10N4O2", "ATP (C10H16N5O13P3)": "C10H16N5O13P3",
    "Buckminsterfullerene (C60)": "C60", "Carbon Nanotube (C)": "C",
    "Glucose (C6H12O6)": "C6H12O6", "Aspirin (C9H8O4)": "C9H8O4",
    "Sodium Chloride (NaCl)": "NaCl", "Sulfuric Acid (H2SO4)": "H2SO4",
    "Hydrochloric Acid (HCl)": "HCl", "Nitric Acid (HNO3)": "HNO3",
    "Phosphoric Acid (H3PO4)": "H3PO4", "Urea (CH4N2O)": "CH4N2O",
    "Paracetamol (C8H9NO2)": "C8H9NO2", "Ibuprofen (C13H18O2)": "C13H18O2"
}

def get_pubchem_compounds(formula: str) -> List[pcp.Compound]:
    """Fetch compounds from PubChem with improved error handling"""
    try:
        return pcp.get_compounds(formula, 'formula')
    except Exception as e:
        st.warning(f"PubChem server error: {str(e)}")
        return []

def prepare_molecule(smiles: str, embed_seed: int = 42) -> Optional[Chem.Mol]:
    """Return a molecule with Hs and valid conformer (3D or 2D)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        # Try 3D conformer generation
        res = AllChem.EmbedMolecule(mol, randomSeed=embed_seed)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            # Fallback to 2D
            AllChem.Compute2DCoords(mol)
        return mol
    except Exception as e:
        st.warning(f"Molecule preparation error: {str(e)}")
        return None
def get_compound_info(formula: str) -> Dict[str, str]:
    """Fetch comprehensive compound data with improved fallback"""
    try:
        # First try our database
        for name, f in CHEMICAL_DATABASE.items():
            if f == formula:
                return {'name': name, 'formula': formula}
        # Then try PubChem
        cmpds = get_pubchem_compounds(formula)
        if cmpds:
            c = cmpds[0]
            return {
                'name': c.iupac_name or 'Unknown',
                'synonyms': ', '.join(c.synonyms[:3]) if c.synonyms else '',
                'formula': c.molecular_formula,
                'pubchem_id': c.cid
            }
        return {'name': 'Unknown', 'formula': formula}
    except Exception as e:
        st.warning(f"Compound lookup failed: {str(e)}")
        return {'name': 'Unknown', 'formula': formula}

from typing import Optional
def speak_response(response):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        male_voice = [v for v in voices if "male" in v.name.lower()]
        engine.setProperty('voice', male_voice[0].id if male_voice else voices[0].id)
        engine.say(response)
        engine.runAndWait()
    except:
        pass
def handle_chem_query(query):
    match_formula = re.search(r"(?:formula|symbol|molecular formula) of ([\w\s\-]+)", query, re.I)
    match_props = re.search(r"(?:properties|structure|weight) of ([\w\s\-]+)", query, re.I)
    name = None
    if match_formula:
        name = match_formula.group(1).strip()
    elif match_props:
        name = match_props.group(1).strip()

    if name:
        try:
            comp = pcp.get_compounds(name, 'name')[0]
            info = []
            if comp.molecular_formula:
                info.append(f"Formula: {comp.molecular_formula}")
            if comp.molecular_weight:
                info.append(f"Mol. Weight: {comp.molecular_weight:.2f} g/mol")
            if comp.iupac_name:
                info.append(f"IUPAC: {comp.iupac_name}")
            return " | ".join(info)
        except:
            return f"❌ Couldn’t find anything about **{name}**."

    # Fallback to OpenRouter
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openrouter/auto",
            "messages": [
                {"role": "system", "content": "You are a smart chemistry tutor. Keep it brief and correct."},
                {"role": "user", "content": query}
            ]
        }
        r = requests.post(url, headers=headers, json=data)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ OpenRouter Error {r.status_code}: {r.json().get('error', {}).get('message', 'Unknown')}"
    except Exception as e:
        return f"❌ Exception: {e}"
from rdkit.Chem import Descriptors

def evaluate_lipinski(mol):
    """Evaluate Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    rules = {
        "molecular_weight < 500": mw < 500,
        "logP < 5": logp < 5,
        "hydrogen_bond_donors ≤ 5": hbd <= 5,
        "hydrogen_bond_acceptors ≤ 10": hba <= 10,
    }

    passed = all(rules.values())
    rules["passed"] = passed
    return rules
def estimate_bioavailability(mol):
    """Estimate oral bioavailability score based on simple heuristics."""
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)

    # Heuristic scoring (simple approximation)
    score = 1.0

    if hbd > 5 or hba > 10:
        score -= 0.3
    if tpsa > 140:
        score -= 0.3
    if rot_bonds > 10:
        score -= 0.2
    if logp < -1 or logp > 5:
        score -= 0.2

    return max(score, 0.0)
def calculate_sascore(mol):
    from sascorer import calculateScore  # make sure `sascorer.py` is in same folder or added to path
    score = calculateScore(mol)
    return round(score, 2)

def get_render_style(style: str, colorscheme: str):
    """
    Map user-selected style + colorscheme to a py3Dmol style dict.
    Notes:
      - 'colorscheme' works reliably for stick/sphere.
      - Cartoon on small molecules does nothing; we still return cartoon, but
        caller can layer a Stick fallback.
      - Surface color left None so py3Dmol uses atomic colors under translucent shell.
    """
    if style == "Stick":
        return {"stick": {"radius": 0.25, "colorscheme": colorscheme}}
    elif style == "Sphere":
        return {"sphere": {"scale": 0.3, "colorscheme": colorscheme}}
    elif style == "Cartoon":
        return {"cartoon": {"color": "spectrum"}}   # ignore colorscheme
    elif style == "Surface":
        return {"surface": {"opacity": 0.9}}        # let atom colors bleed through
    else:
        return {"stick": {"colorscheme": colorscheme}}

@st.cache_data(show_spinner=False)
def formula_to_smiles(query: str) -> Optional[str]:
    """Convert chemical formula or name to SMILES. Uses dictionary and PubChem fallback."""
    smiles_map = {
        "H2O": "O", "CH4": "C", "CH3COOH": "CC(=O)O", "C2H5OH": "CCO",
        "CO2": "O=C=O", "NH3": "N", "NaOH": "[Na+].[OH-]", "KOH": "[K+].[OH-]",
        "HNO3": "O[N+](=O)[O-]", "HCl": "Cl", "H2SO4": "O=S(=O)(O)O", "H3PO4": "OP(=O)(O)O",
        "NaCl": "[Na+].[Cl-]", "CH4N2O": "NC(=O)N",  # Urea
        "C6H12O6": "OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",  # Glucose
        "C8H9NO2": "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
        "C13H18O2": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "C9H8O4": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "C8H10N4O2": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C10H16N5O13P3": "Nc1ncnc2c1ncn2[C@H]3O[C@@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O",  # ATP
        "C60": "C1=CC2=CC3=CC4=CC5=CC6=CC7=CC8=CC9=CC1=C2C3=C4C5=C6C7=C8C9",  # Fullerene (approx.)
        "C2H4": "C=C", "C8H8": "C1=CC=C(C=C1)C=C", "CH3OH": "CO", "C2H6": "CC",
        "C3H8": "CCC", "C4H10": "CCCC", "HCOOH": "O=CO", "NH4OH": "[NH4+].[OH-]",
        "O2": "O=O", "N2": "N#N", "Cl2": "ClCl", "F2": "FF"
    }

    query = query.strip()

    if query in smiles_map:
        return smiles_map[query]

    try:
        compounds = pcp.get_compounds(query, 'name')
        if not compounds:
            compounds = pcp.get_compounds(query, 'formula')
        if compounds and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
    except Exception as e:
        st.warning(f"SMILES conversion failed for '{query}': {e}")

    return None

SUPPORTED_EMT = {"H", "C", "N", "O", "Al", "Na"}

def extract_elements(formula: str) -> set:
    return set(re.findall(r"[A-Z][a-z]?", formula))

def is_quantum_supported(formula: str) -> bool:
    elements = extract_elements(formula)
    return all(e in SUPPORTED_EMT for e in elements)
def show_basic_properties(formula: str):
    st.markdown("### Molecular Formula Summary")

    # Element breakdown
    import re
    from collections import Counter

    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    element_counts = Counter()

    for elem, count in elements:
        count = int(count) if count else 1
        element_counts[elem] += count

    st.write("**Elemental Composition:**")
    for elem, count in element_counts.items():
        st.write(f"- {elem}: {count} atoms")

    # Approximate molecular weight
    try:
        from rdkit.Chem import Descriptors
        smiles = formula_to_smiles(formula)
        mol = prepare_molecule(smiles)
        if mol:
            weight = Descriptors.MolWt(mol)
            st.write(f"**Approximate Molecular Weight:** {weight:.2f} g/mol")
    except Exception as e:
        st.warning(f"Could not calculate molecular weight: {e}")
def show_advanced_analysis(formula: str, style: str, colorscheme: str = "Jmol"):
    smiles = formula_to_smiles(formula)
    if smiles:
        mol = prepare_molecule(smiles)
        if mol:
            try:
                mb = Chem.MolToMolBlock(mol)
                viewer = py3Dmol.view(width=800, height=600)
                viewer.addModel(mb, 'mol')
                viewer.setStyle(get_render_style(style, colorscheme))  # Now works
                viewer.setBackgroundColor("white")
                viewer.zoomTo()
                html = viewer._make_html()
                st.components.v1.html(html, height=600, width=800)
            except Exception as e:
                st.error(f"3D visualization error: {e}")
                st.info("Fallback: showing 2D")
                try:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
                except:
                    st.warning("2D rendering failed.")
        else:
            st.warning("Could not prepare molecule.")
    else:
        st.warning("SMILES conversion failed.")
def calculate_properties(formula: str) -> Dict:
    """Calculate molecular properties with robust conformer handling"""
    comp = Composition(formula)
    elements = {str(e): round(comp.get_atomic_fraction(e)*100, 2) for e in comp.elements}
    props = {}
    smiles = formula_to_smiles(formula)
    
    if smiles:
        mol = prepare_molecule(smiles)
        if mol and mol.GetNumConformers() > 0:
            try:
                props.update({
                    'logp': round(Descriptors.MolLogP(mol), 2),
                    'tpsa': round(Descriptors.TPSA(mol), 2),
                    'mol_weight': round(Descriptors.MolWt(mol), 2),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'h_donors': Descriptors.NumHDonors(mol),
                    'h_acceptors': Descriptors.NumHAcceptors(mol),
                    'formal_charge': Chem.GetFormalCharge(mol)
                })
            except Exception as e:
                st.warning(f"Property calc fallback (2D): {e}")
    
    return {
        'mass': round(comp.weight, 4),
        'elements': elements,
        'properties': props,
        'info': get_compound_info(formula)
    }
def show_3d_molecule(smiles: str, style: str = 'Stick', colorscheme: str = 'Jmol', width: int = 800, height: int = 600) -> None:
    """Display 3D molecule with robust conformer check, dynamic styling, and 2D fallback."""
    mol = prepare_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES — cannot prepare molecule.")
        return

    try:
        # Ensure coordinates exist (fallback to 2D if not)
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)

        mb = Chem.MolToMolBlock(mol)
        if not mb or "V2000" not in mb:
            raise ValueError("Invalid MolBlock generated")

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mb, 'mol')
        viewer.setStyle(get_render_style(style, colorscheme))
        viewer.setBackgroundColor("white")  # You can make this dynamic too
        viewer.zoomTo()

        # Hover labels
        viewer.setHoverable({}, True,
            '''
            function(atom,viewer,event,container) {
                if(!atom.label) {
                    atom.label = viewer.addLabel(
                        "Atom " + atom.serial + ": " + atom.elem,
                        {
                            position: atom,
                            backgroundColor: "black",
                            fontColor: "white",
                            fontSize: 14,
                            inFront: true
                        }
                    );
                }
            }''',
            '''
            function(atom,viewer) {
                if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                }
            }'''
        )

        # Render in Streamlit
        st.components.v1.html(viewer._make_html(), height=height)

    except Exception as e:
        st.error(f"⚠️ 3D visualization error: {str(e)}")
        st.info("Tip: Try switching to 2D view or simplifying the molecule structure.")

        try:
            # Fallback to 2D image
            img = Draw.MolToImage(mol, size=(300, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
        except Exception as e2:
            st.error("2D fallback failed too. Try a simpler molecule.")

import hashlib
# ----------------------
# Cached optimizer (keyed by immutable tuples => safe for Streamlit cache)
# ----------------------
@st.cache_resource
def optimize_positions_cached(pos_tuple: tuple, symbols: tuple, fmax: float = 0.05, steps: int = 200):
    """
    pos_tuple: tuple of (x,y,z) tuples  -- hashable
    symbols: tuple of element symbols     -- hashable
    Returns: (optimized_positions (np.array), forces (np.array), energy_ev (float))
    """
    positions = np.array(pos_tuple, dtype=float)
    ase_atoms = Atoms(list(symbols), positions=positions)
    ase_atoms.set_calculator(EMT())

    # Run BFGS, quiet output
    dyn = BFGS(ase_atoms, logfile=None)
    try:
        dyn.run(fmax=fmax, steps=steps)
    except Exception:
        # If it fails to converge, return current state (no crash)
        pass

    new_positions = ase_atoms.get_positions()
    forces = ase_atoms.get_forces()
    try:
        energy_ev = ase_atoms.get_potential_energy()
    except Exception:
        energy_ev = float("nan")

    return new_positions, forces, energy_ev


# ----------------------
# Main view
# ----------------------
def quantum_calculations_view(formula: str) -> None:
    """Run quantum-style geometry optimization with a safe UI (button + caching)."""

    # --- Parse & validate SMILES / molecule first (no heavy work yet) ---
    smiles = formula_to_smiles(formula)
    if not smiles:
        st.warning("SMILES conversion required for quantum analysis")
        return

    mol = prepare_molecule(smiles)
    if not mol:
        st.warning("Invalid molecule structure")
        return

    if mol.GetNumConformers() == 0:
        st.warning("No 3D conformer available for quantum calculation")
        return

    # --- Basic extracted data (lightweight) ---
    original_coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
    symbols = tuple(atom.GetSymbol() for atom in mol.GetAtoms())
    pos_tuple = tuple(map(tuple, original_coords.tolist()))  # hashable key for caching
    initial_xyz = Chem.MolToXYZBlock(mol)

    # Place Run button in sidebar or main UI - only runs when clicked
    col_run = st.sidebar if st.sidebar else st
    run_now = col_run.button("▶️ Run Quantum Optimization")

    # If we've run before for this molecule, load from session_state to show results quickly
    result_key = "quantum_result_" + hashlib.sha256(initial_xyz.encode()).hexdigest()
    prev = st.session_state.get(result_key, None)

    if run_now or prev is None:
        # If button pressed OR not cached in session_state - call optimizer (cached by content)
        with st.spinner("Running quantum calculations (this may take a little)..."):
            try:
                # Use cached optimizer keyed by positions & symbols
                new_positions, forces, energy_ev = optimize_positions_cached(pos_tuple, symbols, fmax=0.05, steps=200)

                # Build optimized RDKit molecule copy and update coordinates
                optimized_mol = Chem.Mol(mol)
                conf = optimized_mol.GetConformer()
                for i, p in enumerate(new_positions):
                    conf.SetAtomPosition(i, tuple(p))

                optimized_xyz = Chem.MolToXYZBlock(optimized_mol)

                # RMSD
                rmsd = np.sqrt(np.mean(np.sum((original_coords - new_positions) ** 2, axis=1)))

                # Force magnitudes
                force_magnitudes = np.linalg.norm(forces, axis=1)

                # Energy conversion
                try:
                    energy_kj = energy_ev * 96.485
                except Exception:
                    energy_kj = float("nan")

                # Store in session_state so UI doesn't need to recompute on small widget changes
                st.session_state[result_key] = {
                    "initial_xyz": initial_xyz,
                    "optimized_xyz": optimized_xyz,
                    "new_positions": new_positions,
                    "force_magnitudes": force_magnitudes,
                    "energy_ev": energy_ev,
                    "energy_kj": energy_kj,
                    "rmsd": rmsd,
                    "optimized_mol": optimized_mol,
                }
                prev = st.session_state[result_key]

            except Exception as e:
                st.error(f"Quantum calculation failed: {e}")
                st.info("Tip: Try with a simpler molecule or reduce the number of atoms.")
                return

    # If we get here, `prev` contains the results (either from this run or previously cached)
    if prev:
        # --- Metrics row ---
        col1, col2, col3 = st.columns(3)
        with col1:
            ev = prev["energy_ev"]
            kj = prev["energy_kj"]
            st.metric("Optimized Energy", f"{ev:.4f} eV\n({kj:.2f} kJ/mol)")
        with col2:
            st.metric("Number of Atoms", f"{len(symbols)}")
        with col3:
            st.metric("Geometry RMSD", f"{prev['rmsd']:.4f} Å")

        # --- Forces plot (magnitudes) ---
        fig = px.bar(
            x=list(symbols),
            y=prev["force_magnitudes"],
            labels={"x": "Atoms", "y": "Force Magnitude (eV/Å)"},
            title="Atomic Force Distribution (magnitudes)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 3D Viewer ---
        with st.expander("🧪 View Molecular Geometry (Before vs After Optimization)", expanded=True):
            view_option = st.radio("Select structure to view:", ["Before Optimization", "After Optimization"], horizontal=True)
            xyz_to_show = prev["initial_xyz"] if view_option == "Before Optimization" else prev["optimized_xyz"]

            view = py3Dmol.view(width=550, height=420)
            view.addModel(xyz_to_show, "xyz")
            view.setStyle({'stick': {}})
            view.setBackgroundColor("white")
            view.zoomTo()
            st.components.v1.html(view._make_html(), height=420)

        # --- Downloads ---
        st.markdown("### 📥 Download XYZ Files")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️ Before Optimization", prev["initial_xyz"], file_name="before_optimization.xyz")
        with dl2:
            st.download_button("⬇️ After Optimization", prev["optimized_xyz"], file_name="after_optimization.xyz")

        # --- Bond lengths (unique pairs) ---
        with st.expander("📏 Bond Lengths (After Optimization)", expanded=False):
            seen = set()
            opt_mol = prev["optimized_mol"]
            coords = prev["new_positions"]
            for bond in opt_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                pair = tuple(sorted((i, j)))
                if pair in seen:
                    continue
                seen.add(pair)
                ai = opt_mol.GetAtomWithIdx(i).GetSymbol()
                aj = opt_mol.GetAtomWithIdx(j).GetSymbol()
                dist = np.linalg.norm(coords[i] - coords[j])
                st.write(f"{ai}-{aj}: {dist:.3f} Å")

#voice assistant
def chemistry_voice_assistant():
    st.markdown("### 🤖 Ask Chemistry Bot")
    voice_input_enabled = st.toggle("🎙️ Use voice", value=False, key="use_voice_toggle_sidebar")
    if voice_input_enabled:
        if st.button("🎤 Record Question", key="record_btn_sidebar"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                with st.spinner("Listening..."):
                    audio = recognizer.listen(source, timeout=5)
                    try:
                        user_query = recognizer.recognize_google(audio)
                        st.markdown(f"💬 You asked: `{user_query}`")
                        answer = ask_chemistry_bot(user_query)
                        st.markdown(f"**🧠 Assistant:** {answer}")
                        speak(answer)
                    except Exception as e:
                        st.error(f"❌ Could not recognize speech: {e}")
    else:
        user_query = st.text_input("Type your chemistry question:", key="text_input_sidebar")
        if user_query:
            st.markdown(f"💬 You asked: `{user_query}`")
            answer = ask_chemistry_bot(user_query)
            st.markdown(f"**🧠 Assistant:** {answer}")
            speak(answer)

def sidebar_controls() -> tuple:
    """Create simplified sidebar controls"""
    with st.sidebar:
        st.markdown("""
        <div style="background:#f0f2f6; padding:1rem; border-radius:8px;">
            <h3>🚀 About</h3>
            <p>This application was developed by Aqsa Ijaz to provide interactive chemistry visualization and analysis tools.</p>
            <p>Version 5.6</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.title("⚗️ Controls")
        elements = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"
        ]
        selected_elements = st.multiselect(
            "🧩 Build with Elements:", 
            options=elements, 
            help="Select elements to help you build a compound manually."
        )
        if selected_elements:
            st.info(f"Selected elements: {', '.join(selected_elements)}")

        view_mode = st.radio(
            "Analysis Mode:",
            ["🔍 Quick Analysis", "📊 Detailed Report", "🔄 3D Explorer", "🔬 Quantum Calc"],
            index=0
        )
        st.markdown("---")
        chemistry_voice_assistant()
        compound = st.selectbox(
            "Select Compound", 
            sorted(CHEMICAL_DATABASE.keys()),
            index=0
        )
        
        formula = CHEMICAL_DATABASE[compound]
        render_style = None
        bg_color = None
        
        if view_mode == "🔄 3D Explorer":
            style = st.selectbox("🧬 Render Style", ["Stick", "Sphere", "Cartoon", "Surface"])
            bg_color = st.color_picker("Background Color", "#FFFFFF")
            
        return view_mode, compound, formula, render_style, bg_color
    

# Initialize Text-to-Speech engine globally
engine = pyttsx3.init()

def speak(text):
    try:
        if not engine._inLoop:
            engine.say(text)
            engine.runAndWait()
    except RuntimeError as e:
        print(f"[Voice Error]: {e}")

# --- Bot Query Function ---
@st.cache_data(show_spinner=False)
def ask_chemistry_bot(query: str) -> str:
    m_prop = re.search(r'properties of ([\w\s]+)', query, re.I)
    m_form = re.search(r'formula of ([\w\s]+)', query, re.I)
    name = None
    if m_prop:
        name = m_prop.group(1).strip()
    elif m_form:
        name = m_form.group(1).strip()

    if name:
        try:
            comps = pcp.get_compounds(name, 'name')
            if not comps:
                return f"❌ I couldn’t find anything for “{name}” in PubChem."
            c = comps[0]
            formula = c.molecular_formula or 'Unknown'
            mw = getattr(c, 'molecular_weight', None)
            try:
                mw_val = float(mw)
                mw_str = f"{mw_val:.2f} g/mol"
            except (ValueError, TypeError):
                mw_str = "Unknown"

            if m_form:
                return f"The chemical formula for **{name.title()}** is **{formula}**."

            props = [
                f"**Compound:** {name.title()}",
                f"Formula: {formula}",
                f"Molecular Weight: {mw_str}"
            ]
            if c.iupac_name:
                props.append(f"IUPAC Name: {c.iupac_name}")
            if c.synonyms:
                props.append(f"Synonyms: {', '.join(c.synonyms[:3])}")
            return " • ".join(props)
        except Exception as e:
            return f"❌ PubChem lookup failed: {e}"

    return "🤖 I couldn't find a compound name in your question. Try asking like:\n- 'formula of sulfuric acid'\n- 'properties of glucose'"

# Initialize voice engine once at the top
# Instead of defining 'speak()' twice, define once:
engine = pyttsx3.init()
def speak(text):
    try:
        if not engine._inLoop:
            engine.say(text)
            engine.runAndWait()
    except RuntimeError:
        pass
#    """Main application function"""
def main():
    st.markdown("""
        <style>
            .header {
                background: linear-gradient(135deg, #1e88e5, #0d47a1);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .creator-credit {
                color: #00e5ff;
                font-weight: bold;
                font-size: 1.2em;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header">
            <h1 style="text-align:center;">🧪 Molecular Explorer Pro</h1>
            <p style="text-align:center;">
                Interactive Chemistry Analysis Tool — by <span class="creator-credit">Aqsa Ijaz ✨</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load sidebar controls
    view_mode, compound, formula, render_style, bg_color = sidebar_controls()

    # Render different views based on selection
    if view_mode == "🔍 Quick Analysis":
        quick_analysis_view(formula)
    elif view_mode == "📊 Detailed Report":
        detailed_report_view(formula)
    elif view_mode == "🔄 3D Explorer":
        three_d_explorer_view(formula, render_style, bg_color)
    elif view_mode == "🔬 Quantum Calc":
        if is_quantum_supported(formula):
            quantum_calculations_view(formula)
        else:
            st.warning("❌ Quantum calculations not supported for this molecule.")
    from sascorer import readFragmentScores
    readFragmentScores()
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Molecular Explorer Pro v5.6 • Created by Aqsa Ijaz • © 2024
        </div>
    """, unsafe_allow_html=True)

def quick_analysis_view(formula: str) -> None:
    """Display quick analysis view with enhanced visualization"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🧮 Basic Properties")
        results = calculate_properties(formula)
        if results:
            st.metric("Molecular Mass", f"{results['mass']} g/mol")
            if results['elements']:
                fig, ax = plt.subplots(figsize=(6, 6))
                wedges, texts = ax.pie(
                    results['elements'].values(),
                    labels=results['elements'].keys(),
                    startangle=90,
                    textprops={'fontsize': 10}
                )
                ax.axis('equal')
                st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🧪 Compound Info")
        info = get_compound_info(formula)
        st.write(f"**Name**: {info['name'].title()}")
        st.write(f"**Formula**: {info['formula']}")
        smiles = formula_to_smiles(formula)
        # Skip any 3D-related messages completely
st.markdown("</div>", unsafe_allow_html=True)
def detailed_report_view(formula: str) -> None:
    """Display detailed report view with enhanced visualization"""
    tab1, tab2, tab3 = st.tabs(["📈 Properties", "🧪 Composition", "🔬 Advanced"])

    # Tab 1: Detailed Properties
    with tab1:
        st.subheader("📊 Detailed Molecular Properties")
        results = calculate_properties(formula)
        if not results:
            st.warning("No property data available.")
            return

        col1, col2 = st.columns(2)
        with col1:
            fc = results['properties'].get('formal_charge')
            if fc is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>⚡ Formal Charge:</b> {fc}<br>
                        <small style='color:gray;'>Overall net charge</small>
                    </div>
                """, unsafe_allow_html=True)

            logp = results['properties'].get('logp')
            if logp is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>💧 LogP (Hydrophobicity):</b> {logp}<br>
                        <small style='color:gray;'>Higher = fat‑soluble</small>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            tpsa = results['properties'].get('tpsa')
            if tpsa is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>🧲 TPSA (Polar Surface Area):</b> {tpsa} Å²<br>
                        <small style='color:gray;'>Polar interaction potential</small>
                    </div>
                """, unsafe_allow_html=True)

            rotb = results['properties'].get('rotatable_bonds')
            if rotb is not None:
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <b>🔗 Bond Flexibility:</b> {rotb} rotatable bonds<br>
                        <small style='color:gray;'>Higher = more flexible</small>
                    </div>
                """, unsafe_allow_html=True)

    # Tab 2: Elemental Composition (Correctly Placed)
    with tab2:
        st.subheader("🧪 Elemental Composition Analysis")
        results = calculate_properties(formula)
        elems = results.get('elements', {})
        if not elems:
            st.warning("No composition data available.")
            return

        # 🌈 Sparkly Plotly Donut Chart
        fig = go.Figure(data=[go.Pie(
            labels=list(elems.keys()),
            values=list(elems.values()),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=13,
            marker=dict(
                colors=px.colors.qualitative.Pastel,
                line=dict(color='#FFFFFF', width=2)
            )
        )])

        fig.update_layout(
            title={
                'text': "Elemental Composition",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=30, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # 📋 Pretty Composition Table
        df = pd.DataFrame([
            {
                "Symbol": str(el),
                "Count": Composition(formula)[el],
                "Atomic Mass": f"{el.atomic_mass:.4f}",
                "Electronegativity": f"{el.X:.2f}"
            }
            for el in Composition(formula).elements
        ])

        st.markdown("### 📋 Element Summary Table")
        st.dataframe(df.style.background_gradient(cmap="YlGnBu", subset=["Electronegativity"]))

    # Tab 3: Advanced Metrics
    with tab3:
        st.subheader("🔬 Advanced Molecular Analysis")
        smiles = formula_to_smiles(formula)
        if not smiles:
            st.warning("SMILES conversion failed for advanced metrics.")
            return

        mol = prepare_molecule(smiles)
        if mol is None:
            st.warning("Molecule preparation failed for advanced metrics.")
            return

        # Lipinski Rule of 5
        lip = evaluate_lipinski(mol)
        st.markdown("#### ✅ Lipinski Rule of 5")
        for rule, ok in lip.items():
            if rule != "passed":
                st.markdown(f"- {rule.replace('_',' ').title()}: {'✔️' if ok else '❌'}")
        st.success("Passes all Lipinski criteria!" if lip["passed"] else "❌ Does not satisfy all Lipinski criteria")

        # Bioavailability Score
        bio = estimate_bioavailability(mol)
        st.markdown(f"#### 🧬 Bioavailability Score: **{bio:.2f}**")
        st.caption("Higher → more likely to be orally active")

        # Synthetic Accessibility
        try:
            sas = calculate_sascore(mol)
            st.markdown(f"#### 🛠️ Synthetic Accessibility: **{sas:.2f}**")
            st.caption("1 = easy to synthesize, 10 = very difficult")
        except Exception as e:
            st.warning(f"Could not compute SAScore: {e}")

def three_d_explorer_view(formula: str, render_style: str = "Stick", bg_color: str = "#ffffff") -> None:
    """Enhanced 3D molecule explorer with safe styling, hover, auto-rotate, and optional snapshot."""
    smiles = formula_to_smiles(formula)
    if not smiles:
        st.error("❌ Could not get SMILES for 3D view.")
        return

    mol = prepare_molecule(smiles)
    if mol is None:
        st.error("❌ Could not prepare molecule for 3D view.")
        return

    # --- UI Chrome ---
    st.markdown("""
    <style>
    .three-d-container {
        background: linear-gradient(145deg, #f0f2f5, #d9e4f5);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .three-d-title {
        font-size: 1.8rem;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="three-d-container">', unsafe_allow_html=True)
    st.markdown('<div class="three-d-title">🔄 3D Molecule Explorer</div>', unsafe_allow_html=True)

    # --- Controls ---
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        style = st.selectbox("🧬 Style:", ["Stick", "Sphere", "Cartoon", "Surface"],
                             index=["Stick", "Sphere", "Cartoon", "Surface"].index(render_style)
                             if render_style in ["Stick", "Sphere", "Cartoon", "Surface"] else 0)
    with col2:
        colorscheme = st.selectbox("🎨 Colorscheme:", ["default", "Jmol", "chain", "element"], index=1)
    with col3:
        bg_color = st.color_picker("🖼️ Background Color:", value=bg_color)

    col4, col5 = st.columns([1, 1])
    with col4:
        rotate = st.toggle("🔄 Auto-Rotate", value=True)
    with col5:
        enable_snapshot = st.toggle("📸 Enable Snapshot Button", value=False)

    # --- Build 3D View ---
    try:
        mb = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=700, height=500)
        viewer.addModel(mb, 'mol')

        # Main style
        main_style = get_render_style(style, colorscheme)
        viewer.setStyle(main_style)

        # If cartoon selected but nothing would render, layer stick fallback
        if style == "Cartoon":
            viewer.addStyle({}, {"stick": {"radius": 0.15, "colorscheme": colorscheme}})

        viewer.setBackgroundColor(bg_color)
        viewer.zoomTo()

        if rotate:
            viewer.spin(True)

        if enable_snapshot:
            viewer.addButton("📸 Snapshot", """
                function(){
                  viewer.render();
                  viewer.downloadImage({format:'png', backgroundColor:'white'});
                }
            """)

        # Hover labels
        viewer.setHoverable({}, True,
            '''
            function(atom,viewer,event,container) {
                if(!atom.label) {
                    atom.label = viewer.addLabel(
                        "Atom " + atom.serial + ": " + atom.elem,
                        {
                            position: atom,
                            backgroundColor: "black",
                            fontColor: "white",
                            fontSize: 14,
                            inFront: true
                        }
                    );
                }
            }''',
            '''
            function(atom,viewer) {
                if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                }
            }'''
        )

        st.components.v1.html(viewer._make_html(), height=500)

    except Exception as e:
        st.error(f"Rendering failed: {e}")
        st.info("Trying 2D view as fallback...")
        try:
            img = Draw.MolToImage(mol, size=(300, 300))
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
        except Exception:
            st.error("2D fallback failed too.")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
