# Import required libraries
import openai 
import pyttsx3 
import speech_recognition as sr 
import re
import os
import time
import datetime
from io import StringIO, BytesIO
import base64
from typing import Dict, Optional, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from pymatgen.core.composition import Composition 
import pubchempy as pcp
import py3Dmol 
from ase import Atoms 
from ase.calculators.emt import EMT 
from ase.optimize import BFGS 
import streamlit as st
from streamlit.components.v1 import html as st_html 
import openai 
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

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

RENDER_STYLES = {
    "Stick": {'stick': {'radius': 0.15, 'colorscheme': 'cyanCarbon'}},
    "Sphere": {'sphere': {'scale': 0.25, 'colorscheme': 'greenCarbon'}},
    "Cartoon": {'cartoon': {'color': 'spectrum'}},
    "Surface": {
        'cartoon': {},
        'surface': {'type': 'vdw', 'opacity': 0.7, 'colorscheme': 'whiteCarbon'}
    }
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

import pubchempy as pcp
from typing import Optional

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
def show_advanced_analysis(formula: str, style: str):
    smiles = formula_to_smiles(formula)
    if smiles:
        mol = prepare_molecule(smiles)
        if mol:
            try:
                mb = Chem.MolToMolBlock(mol)
                viewer = py3Dmol.view(width=800, height=600)
                viewer.addModel(mb, 'mol')
                viewer.setStyle(RENDER_STYLES.get(style, RENDER_STYLES['Stick']))
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
                    'num_atoms': mol.GetNumAtoms(),
                    'mol_weight': round(Descriptors.MolWt(mol), 2),
                    'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'num_h_donors': Descriptors.NumHDonors(mol),
                    'num_h_acceptors': Descriptors.NumHAcceptors(mol)
                })
            except Exception as e:
                st.warning(f"Property calc fallback (2D): {e}")
    
    return {
        'mass': round(comp.weight, 4),
        'elements': elements,
        'properties': props,
        'info': get_compound_info(formula)
    }

def show_3d_molecule(smiles: str, style: str = 'Stick', width: int = 800, height: int = 600) -> None:
    """Display 3D molecule with robust conformer and fallback."""
    mol = prepare_molecule(smiles)
    if mol is None:
        st.error("Invalid SMILES, cannot prepare molecule.")
        return

    try:
        # Force 2D if 3D fails
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        
        mb = Chem.MolToMolBlock(mol)
        if not mb or "V2000" not in mb:
            raise ValueError("Invalid MolBlock generated")

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mb, 'mol')
        viewer.setStyle(RENDER_STYLES.get(style, RENDER_STYLES['Stick']))
        viewer.zoomTo()
        viewer_html = viewer.html()
        st.components.v1.html(viewer_html, height=height, width=width)

    except Exception as e:
        st.error(f"3D visualization error: {str(e)}")
        st.info("Tip: Try switching to 2D view or simplifying the molecule structure")

        try:
            # Fallback to 2D image
            img = Draw.MolToImage(mol, size=(300, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(f'<img src="image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
        except Exception as e2:
            st.error("2D fallback failed too. Try a simpler molecule.")

def quantum_calculations_view(formula: str) -> None:
    """Quantum chemistry calculations with conformer safety"""
    st.subheader("🔬 Quantum Mechanics Analysis")
    try:
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
        
        coords = mol.GetConformer().GetPositions()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        ase_atoms = Atoms(atoms, positions=coords)
        ase_atoms.set_calculator(EMT())
        dyn = BFGS(ase_atoms)
        dyn.run(fmax=0.05)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimized Energy", f"{ase_atoms.get_potential_energy():.4f} eV")
        with col2:
            st.metric("Number of Atoms", f"{len(ase_atoms)}")
            
        fig = px.bar(
            x=atoms,
            y=ase_atoms.get_forces().sum(axis=1),
            labels={"x": "Atoms", "y": "Force Magnitude (eV/Å)"},
            title="Atomic Force Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Quantum calculation failed: {str(e)}")
        st.info("Tip: Try with a simpler molecule")

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
            render_style = st.selectbox("Render Style", list(RENDER_STYLES.keys()))
            bg_color = st.color_picker("Background Color", "#FFFFFF")
            
        return view_mode, compound, formula, render_style, bg_color
    
def chemistry_voice_assistant():
    st.markdown("## 🎤 Chemistry Voice Assistant")
    
    if st.button("Start Listening"):
        recognizer = sr.Recognizer()
        engine = pyttsx3.init()

        try:
            with sr.Microphone() as source:
                st.info("🎙 Listening...")
                audio = recognizer.listen(source, timeout=5)
                st.info("🔍 Processing...")
                query = recognizer.recognize_google(audio)
                st.success(f"You asked: {query}")

                # ✅ NEW GPT CALL (3.5 turbo instead of 4)
                client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a chemistry tutor."},
                        {"role": "user", "content": query}
                    ]
                )
                answer = response.choices[0].message.content
                st.markdown(f"**🧠 Assistant:** {answer}")

                # 🔊 Speak response
                engine.say(answer)
                engine.runAndWait()

        except sr.WaitTimeoutError:
            st.warning("⏱️ Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.warning("🤔 Couldn’t understand. Try speaking clearly.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def main() -> None:
    """Main application function"""
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
            color: #ff6d00;
            font-weight: bold;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header">
        <h1 style="color:white; text-align:center;">🧪 Molecular Explorer Pro</h1>
        <p style="text-align:center; font-size:1.1rem;">
            Advanced Chemistry Analysis Suite • <span class="creator-credit">Created by Aqsa Ijaz</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    view_mode, compound, formula, render_style, bg_color = sidebar_controls()
    
    if view_mode == "🔍 Quick Analysis":
        quick_analysis_view(formula)
    elif view_mode == "📊 Detailed Report":
        detailed_report_view(formula)
    elif view_mode == "🔄 3D Explorer":
        three_d_explorer_view(formula, render_style, bg_color)
    elif view_mode == "🔬 Quantum Calc":
     if view_mode == "🔬 Quantum Calc":
      if is_quantum_supported(formula):
        quantum_calculations_view(formula)
    else:
        st.warning("❌ Quantum calculations not supported for this molecule.")
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
        if smiles:
            show_3d_molecule(smiles, width=400, height=300)
        else:
            st.warning("3D preview not available for this compound")
        st.markdown("</div>", unsafe_allow_html=True)

def detailed_report_view(formula: str) -> None:
    """Display detailed report view with enhanced visualization"""
    tab1, tab2, tab3 = st.tabs(["📈 Properties", "🧪 Composition", "🔬 Advanced"])
    
    with tab1:
        st.subheader("Detailed Molecular Properties")
        results = calculate_properties(formula)
        if results:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Molecular Mass", f"{results['mass']} g/mol")
                if 'logp' in results['properties']:
                    st.metric("LogP", f"{results['properties']['logp']}")
            with col2:
                if 'tpsa' in results['properties']:
                    st.metric("TPSA", f"{results['properties']['tpsa']} Å²")
                if 'mol_weight' in results['properties']:
                    st.metric("Molecular Weight", f"{results['properties']['mol_weight']} g/mol")
    
    with tab2:
        st.subheader("Elemental Composition Analysis")
        results = calculate_properties(formula)
        if results and results['elements']:
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                results['elements'].values(),
                labels=results['elements'].keys(),
                autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
                startangle=90,
                textprops={'fontsize': 12},
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
            )
            ax.legend(
                wedges,
                [f"{k} ({v}%)" for k, v in results['elements'].items()],
                title="Elements",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            ax.axis('equal')
            st.pyplot(fig)
            
            elements = []
            for el in Composition(formula).elements:
                elements.append({
                    'Symbol': str(el),
                    'Amount': Composition(formula)[el],
                    'Atomic Mass': f"{el.atomic_mass:.4f}",
                    'Electronegativity': f"{el.X:.2f}",
                    'Group': el.group,
                    'Period': el.row
                })
            st.dataframe(
                pd.DataFrame(elements).style.background_gradient(cmap='YlOrBr', subset=['Electronegativity']),
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Advanced Molecular Analysis")
    smiles = formula_to_smiles(formula)
    if smiles:
        mol = prepare_molecule(smiles)
        if mol is not None:
            try:
                mb = Chem.MolToMolBlock(mol)
                viewer = py3Dmol.view(width=800, height=600)
                viewer.addModel(mb, 'mol')
                viewer.setStyle(RENDER_STYLES['Stick'])  # Default style
                viewer.setBackgroundColor("#ffffff")
                viewer.zoomTo()
                viewer_html = viewer._make_html()
                st.components.v1.html(viewer_html, height=600, width=800)
            except Exception as e:
                st.error(f"3D visualization error: {e}")
                st.info("Fallback: showing 2D")
                img = Draw.MolToImage(mol, size=(300, 300))
                buf = BytesIO()
                img.save(buf, format="PNG")
                img_str = base64.b64encode(buf.getvalue()).decode()
                st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)
        else:
            st.warning("Could not prepare molecule.")
    else:
        st.warning("SMILES conversion failed.")

def three_d_explorer_view(formula: str, render_style: str, bg_color: str) -> None:
    """Enhanced 3D explorer with robust error handling and graceful fallbacks."""
    col1, col2 = st.columns([1, 2])

    # Sidebar panel
    with col1:
        st.subheader("🎨 Visualization Options")
        st.write(f"**Viewing**: `{formula}`")
        st.write(f"**Render Style**: `{render_style}`")
        st.write(f"**Background Color**: `{bg_color}`")

    # Molecule Viewer
    with col2:
        smiles = formula_to_smiles(formula)
        if not smiles:
            st.warning("🚫 SMILES conversion failed. Cannot visualize this formula.")
            return

        mol = prepare_molecule(smiles)
        if mol is None:
            st.warning("🚫 Molecule preparation failed. Structure invalid or incomplete.")
            return

        try:
            mb = Chem.MolToMolBlock(mol)
            if not mb or "V2000" not in mb:
                raise ValueError("Invalid MolBlock generated from molecule")

            viewer = py3Dmol.view(width=500, height=400)
            viewer.addModel(mb, 'mol')
            viewer.setStyle(RENDER_STYLES.get(render_style, RENDER_STYLES['Stick']))
            viewer.setBackgroundColor(bg_color)
            viewer.zoomTo()

            viewer_html = viewer._make_html()
            st.components.v1.html(viewer_html, height=400, width=500)

        except Exception as e:
            st.error(f"🧪 3D visualization error: `{e}`")
            st.info("💡 Tip: Try switching to 2D view or simplify the molecule.")

            # 2D fallback
            try:
                img = Draw.MolToImage(mol, size=(300, 300))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                st.markdown(f'<img src="data:image/png;base64,{img_str}" width="300">', unsafe_allow_html=True)

            except Exception as e2:
                st.error("❌ 2D structure rendering also failed.")
                st.text(f"Reason: {e2}")
if __name__ == "__main__":
    main()