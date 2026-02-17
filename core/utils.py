"""
utils.py

This module contains utility functions and classes used throughout the retrosynthesis project.
These include functions for SMILES canonicalization, database checks, weight normalization,
and random selection, as well as helper classes like `Availability` and `Reaction`. It also 
contains the `Retrosim` class, which implements the single-step retrosynthesis model from
RDEnzyme and RetroBioCat.
"""

import sqlite3
import random
from typing import List, TypeVar, Tuple
from dataclasses import dataclass
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from template_extractor import extract_from_reaction
from main_v3 import rdchiralRun
import numpy as np
#from scorer import SCScorer
from enum import Enum

class Availability(Enum):
    """
    Enum for availability of a chemical compound.
    Used to classify compounds as buyable, synthesizable, or unavailable.
    """
    BUYABLE = 0
    SYNTHESIZABLE = 1
    NONE = 2

class Retrosim:
    def __init__(self, reference_data_path='rhea_atom_mapped_for_jx_cache.pkl', retrobiocat_path='data/retrobiocat_database.pkl'):
        """
        Initialize RDEnzyme for similarity-based retrosynthesis analysis.
        
        Parameters:
            reference_data_path (str): Path to RDEnzyme reference reaction database pickle file
            retrobiocat_path (str): Path to RetroBioCat reference reaction database pickle file
            scscore_path (str): Path to model used for SCScore function
        
        """

        # Load data for RDEnzyme and RetroBioCat
        self.reference_data:pd.DataFrame = pd.read_pickle(reference_data_path)
        self.retrobiocat_reference_data:pd.DataFrame = pd.read_pickle(retrobiocat_path)
        
        # Create the cache for each tool
        self.jx_cache = self.create_rdenzyme_cache()
        self.template_cache = self.create_retrobiocat_cache() 
        
        
    def create_rdenzyme_cache(self):
        """
        Generate a reaction template cache for retrosynthesis analysis via RDEnzyme
        """
        
        jx_cache = {}
        
        for jx in self.reference_data.index:
            reaction_smarts = self.reference_data.at[jx, 'reaction_smarts']
            
            if not reaction_smarts:
                continue
                
            try:
                rxn = rdchiralReaction(reaction_smarts)
                rcts_ref_fp = self.reference_data.at[jx, 'rcts_ref_fp']
                
                if rcts_ref_fp is None:
                    continue
                    
                jx_cache[jx] = (rxn, rcts_ref_fp)
                
            except Exception as e:
                print(f"Error creating RDChiral reaction for index {jx}: {str(e)}")
                
        return jx_cache
    
    def create_retrobiocat_cache(self):
        """
        Generate a reaction template cache for retrosynthesis analysis via RetroBioCat
        """
        template_cache = {}

        for idx, name, rxn_smarts, rxn_type in self.retrobiocat_reference_data.itertuples():
            if rxn_smarts in template_cache:
                continue
            else:
                rxn = rdchiralReaction(rxn_smarts)
                template_cache[rxn_smarts] = rxn
                
        return template_cache
                      
    @staticmethod
    def calculate_fingerprint(smiles):
        """
        Calculate Morgan fingerprint for a given SMILES string.
        """
        fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2, useChirality=True, useFeatures=True)
        return fp 

    def RetroBioCat(self, prod_smiles, prod_fp) -> List[Tuple[str, str, str, float]]:
        """
        Performs retrosynthetic analysis on a target molecule using RetroBioCat templates
        """
        prod = rdchiralReactants(prod_smiles)

        # Results storage list
        results = []
        # Loop through the template set
        for idx, name, rxn_smarts, rxn_type in self.retrobiocat_reference_data.itertuples():
            # Check if template is already in cache for direct application
            if rxn_smarts in self.template_cache:
                rxn = self.template_cache[rxn_smarts]
                
            else:
                #print(f"NOT IN CACHE!!!!RBC")
                # If not in cache, create and store it
                rxn = rdchiralReaction(rxn_smarts)
                self.template_cache[rxn_smarts] = rxn
            try:
                # Apply the template
                outcomes = rdchiralRun(rxn, prod, combine_enantiomers=False)

                for precursors in outcomes:
                    results.append((name, precursors, rxn_smarts))
            except Exception as e:
                #print(f"RBC Issue: {e}")
                continue
                
        return results     
    
    def single_step_retro(self, target_molecule, max_precursors=20, debug=True, retrobiocat=False) -> List[Tuple[str, str, str, str, float, float]]:
        """
        Perform single-step retrosynthesis analysis on the target molecule using RDEnzyme similarity-based method.
        
        Parameters:
        target_molecule (str): SMILES string of the target molecule to analyze
        max_precursors (int, default=50): Maximum number of similar reactions to consider from the reference database
        debug (bool, default=True): If True, prints detailed information about the analysis process
        retrobiocat (bool, default=False): If True, includes additional predictions from RetroBioCat in the results
          
        """
    
        rct = rdchiralReactants(target_molecule)
        
        if debug:
            print(f"Analyzing product: {target_molecule}") 
        
        # Calculate similarities
        fp = self.calculate_fingerprint(target_molecule)
        sims = DataStructs.BulkDiceSimilarity(fp, [fp_ for fp_ in self.reference_data['prod_fp']])
        
        # Sort similarity metric in the reverse order, from most to least similar
        js = np.argsort(sims)[::-1]
        probs = {}
        rhea_id = {}
        smarts = {}
        
        # Look into each similar rxn in js
        for ji, j in enumerate(js[:max_precursors]):
            jx = self.reference_data.index[j]
            current_rhea_id = self.reference_data['id'][jx]
            current_smarts = self.reference_data.at[jx, 'reaction_smarts']
            sims_j = sims[j]
            
            if debug and ji < 5:
                print(f"\nPrecedent {ji+1}")
                print(f"Similarity score: {sims[j]}")
                print(f"Reference reaction: {self.reference_data['rxn_smiles'][jx]}")
            
            if jx in self.jx_cache:
                (rxn, rcts_ref_fp) = self.jx_cache[jx]              
            else:
                try:
                    #print("NOT IN CACHE!!!!RD")
                    rxn_smiles = self.reference_data['rxn_smiles'][jx]

                    rxn_smiles = rxn_smiles[0]
                    rct_0, rea_0, prd_0 = rxn_smiles.split(' ')[0].split('>')
                    
                    # Extract template
                    reaction = {'reactants': rct_0,'products': prd_0,'_id': self.reference_data['id'][jx]}
                    template = extract_from_reaction(reaction)

                    #Load into rdChiralReaction
                    rxn = rdchiralReaction(template['reaction_smarts'])

                    #get the reactants to compute reactant fingerprint
                    prec_rxn = self._get_precursor_goal(rxn_smiles)
                    
                    # get rcts reference fingerprint
                    rcts_ref_fp = self.calculate_fingerprint(prec_rxn)

                    #Save for future use
                    #print("Saved template in cache")
                    self.jx_cache[jx] = (rxn, rcts_ref_fp)

                    if debug and ji < 5:
                        print(f"Template: {template['reaction_smarts']}")
                except:
                    #print("Could not save the template")
                    continue
            
            try:
                # Run retrosynthesis
                outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
            except Exception as e:
                print(e)
                outcomes = []

            if debug and ji < 5:
                print(f"Number of outcomes: {len(outcomes)}")

            if outcomes:
                outcome_fps = [self.calculate_fingerprint(precursors) for precursors in outcomes]
                precursors_sims = DataStructs.BulkDiceSimilarity(rcts_ref_fp, outcome_fps)

                for precursors, precursors_sim in zip(outcomes, precursors_sims):
                    overall_score = precursors_sim * sims_j
                    smarts[precursors] = current_smarts

                    if precursors in probs:
                        probs[precursors] = max(probs[precursors], overall_score)
                    else:
                        probs[precursors] = overall_score
                        rhea_id[precursors] = current_rhea_id

                
            # Process outcomes
            #for precursors in outcomes:
            #    precursors_fp = self.calculate_fingerprint(precursors)
            #    precursors_sim = DataStructs.BulkDiceSimilarity(precursors_fp, [rcts_ref_fp])[0]

                #overall_score = precursors_sim * sims[j]
                #smarts[precursors] = self.reference_data.at[jx, 'reaction_smarts']

                # If this precursor structure was already found through a different template/reaction
                #if precursors in probs:
                #    probs[precursors] = max(probs[precursors], overall_score)
                #else:
                #    probs[precursors] = overall_score
                #    rhea_id[precursors] = current_rhea_id

                if debug and ji < 5:
                    print(f"Found precursor: {precursors}")
                    print(f"Score: {overall_score}")
        
        ranked_output:List[Tuple[str, str, str, str, float]] = []
        
        for prec, prob in sorted(probs.items(), key=lambda x:x[1], reverse=True):
            ranked_output.append((
                "RHEA", # Source
                rhea_id[prec],  # RHEA ID
                smarts[prec], # Reaction SMARTS
                prec,  # precursor SMILES
                prob # probability / similarity to precedent
            ))
            
        
        if retrobiocat:
            retrobiocat_output = self.RetroBioCat(target_molecule, fp)
            if retrobiocat_output:
                for name, prec, rxn in retrobiocat_output:
                    ranked_output.append((
                        "RetroBioCat", # Source
                        name, # Reaction name
                        rxn, # Reaction SMARTS
                        prec, # precursor SMILES
                        1
                    ))            

        return ranked_output
    
    def _get_precursor_goal(self, rxn_smiles):
        """
        Extract and process precursor goal from reaction SMILES.
        """
        if isinstance(rxn_smiles, list):
            rxn_smiles = rxn_smiles[0]
        reactants = rxn_smiles.split('>')[0]
        prec_goal = Chem.MolFromSmiles(reactants)
        [a.ClearProp('molAtomMapNumber') for a in prec_goal.GetAtoms()]
        prec_goal = Chem.MolToSmiles(prec_goal, True)
        return Chem.MolToSmiles(Chem.MolFromSmiles(prec_goal), True)
    
def canonicalize_smiles(smi:str) -> str:
    """
    Canonicalize a SMILES string to ensure a consistent representation of the molecule.
    If canonicalization fails, the original SMILES string is returned.

    :param smi: SMILES string to be canonicalized
    :return: Canonicalized SMILES string or the original string if an error occurs
    """
    try:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except:
        print(f"ERROR: Cannot Canonicalize {smi}")
        canon_smi = smi
    return canon_smi

def neutralize_smiles(smiles):
    """Neutralize charged atoms in a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)

def check_in(smile:str, cursor:sqlite3.Cursor, table_name:str) -> bool:
    """
    Check if a SMILES string exists in the database.

    :param smile: SMILES string to check
    :param cursor: SQLite cursor object
    :param table_name: Name of the table to check in
    :return: True if the SMILES string exists in the table, False otherwise
    """
    # Execute an SQL query to check if SMILES is in the specified table
    cursor.execute(f'SELECT 1 FROM {table_name} WHERE SMILES = ?', (smile,))
    result = cursor.fetchone() # Get first row from query result (or None if no rows match)
    return bool(result)

def check_available(smiles:str, buyables:sqlite3.Cursor, excluded:sqlite3.Cursor, custom:sqlite3.Cursor=None) -> Availability:
    """
    Check the availability of a SMILES string. The function checks the following in order:
    1. If the compound is in the `excluded` table, it is considered unavailable (NONE).
    2. If the compound is in the `buyables` table, it is considered buyable (BUYABLE).
    3. If the compound is in the `custom` table, it is considered  buyable (BUYABLE).

    :param smiles: SMILES string to check
    :param buyables: SQLite cursor object for the buyables table
    :param excluded: SQLite cursor object for the excluded table
    :param custom: SQLite cursor object for the custom table
    :return: Availability enum value indicating the availability status
    """
    if check_in(smiles, excluded, 'excluded'):
        return Availability.NONE
    
    def check_buyability(smi:str) -> bool:
        if custom is not None and check_in(smi, custom, 'buyable'):
            return True
        return check_in(smi, buyables, 'buyable')
    
    # Check original SMILES
    if check_buyability(smiles):
        return Availability.BUYABLE
    
    # Quick fix - check neutralized molecule as well - this is due to differences in the buyable DB
    if check_buyability(neutralize_smiles(smiles)):
        return Availability.BUYABLE
        
    return Availability.NONE
