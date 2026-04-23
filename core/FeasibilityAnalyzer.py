import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.RDLogger as RDLogger
from typing import List, Tuple
RDLogger.DisableLog('rdApp.*')  # Silence RDKit warnings
from collections import Counter
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from ReactChemChecker import *
import os
import pickle
from tree_utils import draw_reaction_with_symbols
import sqlite3
from tqdm import tqdm

class FeasibilityAnalyzer:
    def __init__(self, fp_pickle_path="../data/02152026_directionality_dataset.pkl"):
        
        self.df = pd.read_pickle(fp_pickle_path)
        self.query_template_cache = {}
        self.query_result_cache = {}

    def extract_query_template(self, query_can, debug=False):
        """Extract and cache forward reaction rule from query."""
        if query_can in self.query_template_cache:
            return self.query_template_cache[query_can]
        
        # extract template for query
        query_template = template_extractor(query_can, debug=debug)

        self.query_template_cache[query_can] = query_template
        return query_template
        
    def get_final_cand_list(self, query_rxn_fp, query_prod_fp, debug=False):
        """Get the top 500 most similar reactions to the query."""

        if debug:
            print(f"Using {len(self.df)} candidate reactions")

        # Compute similarities with each candidate reaction in the dataset
        sims = []
        for jx, row_j in self.df.iterrows():
            cand_rxn_fp = row_j['rxn_fp']
            cand_prod_fp = row_j['prod_fp']
            if cand_rxn_fp is None or cand_prod_fp is None:
                continue

            rxn_sim = DataStructs.TanimotoSimilarity(query_rxn_fp, cand_rxn_fp)
            prod_sim = DataStructs.TanimotoSimilarity(query_prod_fp, cand_prod_fp)
            overall_sim = rxn_sim * prod_sim
            sims.append((jx, overall_sim, rxn_sim, prod_sim))

        if debug:
            print(f"Computed similarities for {len(sims)} candidate reactions")

        # Sort candidates by similarity (with highest first)
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True) 

        return sims_sorted[:500]
    
    def _evaluate_candidates(self, candidates, top_n, debug=False):
        """Get the feasibility label from the top most similar reaction with matching chemistry to the query."""
        
        if not candidates:
            return None
        
        # Get top 1 Feasibility Marker
        top1_marker = None

        for ix, cand in enumerate(candidates[:top_n]):
            cand_idx, sim_val, rxn_sim, prod_sim = cand
            cand_feas = self.df.loc[cand_idx].get('feasible?', 'N/A')

            if not top1_marker:
                top1_marker = cand_feas

            if debug:
                print(f"Cand {ix}: RHEA ID {self.df.loc[cand_idx].get('RHEA_ID', 'N/A')}")
                print(f"SMILES: {self.df.loc[cand_idx].get('can_rxn_smiles', 'N/A')}")
                print(f"Feasibility: {self.df.loc[cand_idx].get('feasible?', 'N/A')}")
                print(f"Overall Sim: {sim_val}; Rxn Sim: {rxn_sim}; Prod Sim: {prod_sim}")

    
        if debug:
            print(f"===\nFeasible?: {top1_marker}")

        return top1_marker
    
    def _apply_templates(self, candidates, query_can, top_n, debug=False):
        """Apply reaction templates to candidates and evaluate results."""

        template_matches = [] 
        top1_template_found = False 
        
        if debug: 
            print(f"\nStarting template application with {len(candidates)} candidates") 
            
        # Get query template 
        query_template = self.extract_query_template(query_can) 
        
        # Apply the template to top N 
        for i, (cidx, sim_score, rxn_sim, prod_sim) in enumerate(candidates):
            if len(template_matches) == top_n:
                if debug:
                    print(f"Reached {top_n} template matches. Stopping early at candidate {i}")
                break
            
            c_smi = self.df.loc[cidx].get('can_rxn_smiles', self.df.loc[cidx].get('rxn_smiles'))
            c_feas = self.df.loc[cidx].get('feasible?', None)
            
            if debug: 
                print(f"\nCandidate {i+1}: Applying template to {c_smi}\nFeasibility marker: {c_feas}")
            
            try:
                temp_app, method = reaction_chemistry_checker(query_template, c_smi, debug=debug)
                if debug:
                    print(f"Template applied?: {temp_app}")
                
                if temp_app:
                    if not top1_template_found:
                        top1_template_found = True
                        if debug:
                            print(f"First template match found at candidate {i+1}")
                    
                    template_matches.append((cidx, sim_score, rxn_sim, prod_sim))
                        
                else:
                    if debug:
                        print(f"Template not applied to candidate {i+1}")
                        
            except Exception as e:
                if debug:
                    print(f"Template application failed: {str(e)}")
                continue
        
        if debug:
            print(f"Template application completed. Found {len(template_matches)} matches")
            print(f"Top1 template found: {top1_template_found}")

        if template_matches:
            # Evaluate template matches
            result = self._evaluate_candidates(template_matches, top_n, debug=debug)
        else:
            result = "Unknown"
        
        # Add to result cache
        self.query_result_cache[query_can] = result
        
        return result
    
    def feasibility_predictor(self, query_smi, top_n=1, debug=False):

        query_can = canonicalize_reaction(query_smi)
        query_rxn_fp, query_prod_fp = compute_reaction_fp(query_can)
        if query_rxn_fp is None and query_prod_fp is None:
            if debug:
                print("Error: Could not compute fingerprints for the input SMILES")
            return None

        if debug:
            print("Analyzing user input reaction:")
            print(f"SMILES: {query_can}")

        # check if feasibility of query was already predicted before
        if query_can in self.query_result_cache:
            return self.query_result_cache[query_can]
        
        cand_list = self.get_final_cand_list(query_rxn_fp, query_prod_fp, debug=debug)
        return self._apply_templates(cand_list, query_can, top_n, debug=debug)
    

    #### Single-step planner processing ####
    
    def single_step_feasibility(self, result:list, smiles:str, debug:bool = False) -> Tuple[List, List]:
        """Process outcomes generated by single-step planner and checks feasibility"""
        if not result:
            return [], [], []
        
        feas_rxn = []
        not_feas_rxn = []
        unknown_feas_rxn = []
        
        for item in result:
            if item[0] == 'RHEA':
                proposed_rxn = add_missing_components(rhea_id=item[1], precursors=item[3].split('.'), target=smiles, debug=debug)
                feas = self.feasibility_predictor(proposed_rxn, top_n=1, debug=debug)
                if feas == 'Yes':
                    feas_rxn.append(item)
                elif feas == "Unknown":
                    unknown_feas_rxn.append(item)
                else:
                    not_feas_rxn.append(item)         
            else: # RetroBioCat
                feas_rxn.append(item)

        return (feas_rxn, not_feas_rxn, unknown_feas_rxn)
    
    ########################################
    
    #### Multi-step planner processing ####

    def check_step_feasibility(self, reaction_info, debug=False):
        """Process single step from pathway"""

        source = reaction_info['source']
        reaction_name = reaction_info['reaction_name']

        if source == 'RetroBioCat':
            feasibility = "Yes"

        else:
            # For RHEA entries, we need to determine feasibility 
            try:
                proposed = add_missing_components(rhea_id=reaction_name, precursors=reaction_info['precursor_smiles'], target=reaction_info['parent_smiles'])
                if debug:
                    print(f"  Added missing components: {proposed}")

                # Get feasibility prediction
                feas_result = self.feasibility_predictor(proposed, temp_app=True, top_n=1, debug=debug)

                if debug:
                    print(f"  feasibility: {feas_result}")

                feasibility = feas_result

            except Exception as e:
                if debug:
                    print(f"Error processing entry {reaction_name}: {e}")
                feasibility = "Unknown"

        return feasibility

    def analyze_pathway_feasibility(self, pathway_root, debug=False):

        path_steps = traverse_pathway_tree(pathway_root)
        feasibility_markers = []
        unfeasible_steps = []
        
        for i, reaction in enumerate(path_steps):
            feas = self.check_step_feasibility(reaction_info=reaction, debug=debug)    
            feasibility_markers.append(feas)

            if debug:
                print(f"Reaction: {reaction['reaction_name']} ({reaction['source']})")
                print(f"  Feasibility: {feas}")
                
            if feas != "Yes":
                unfeasible_steps.append({
                    'step_number': i + 1,
                    'reaction_name': reaction['reaction_name'],
                    'feasibility': feas})

        # Check if all steps are feasible
        is_feasible = all(feasibility == "Yes" for feasibility in feasibility_markers)

        if is_feasible:
            return pathway_root, "feasible", []
        else:
            return pathway_root, "not feasible", unfeasible_steps
        
    def process_paths_sample(self, paths, debug=False):

        feasible_paths = []
        unfeasible_paths = []
        for path in paths:
            try:
                root, feas, unfeasible_steps = self.analyze_pathway_feasibility(path, debug=debug)

                if feas == "feasible":
                    feasible_paths.append((root, None))

                else:
                    unfeasible_paths.append((root, unfeasible_steps))

            except Exception as e:
                print(f"Error with pathway: {e}")

        return feasible_paths, unfeasible_paths

    def parallel_feasibility_pathway_analysis(self, paths, saved_file="test_data", n_processes=None):

        # Get number of processes
        if n_processes is None:
            try:
                n_processes = len(os.sched_getaffinity(0))
            except Exception as e:
                #print(e)
                n_processes = 1

        # split data
        sample_size = len(paths) // n_processes
        samples = []
        for i in range(n_processes):
            start_idx = i * sample_size
            if i == n_processes - 1:  # Last chunk gets remaining data
                end_idx = len(paths)
            else:
                end_idx = (i + 1) * sample_size
            samples.append(paths[start_idx:end_idx])

        # partial function with analyzer
        process_func = self.process_paths_sample

        # process samples in parallel
        print(f"Processing {len(paths)} pathways using {n_processes} processes")
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, samples),
                total=len(samples),
                desc="Processing samples"
            ))

        feasible_pathways = []
        not_feasible_pathways = []
        for feasible_results, not_feasible_results in results:
            feasible_pathways.extend(feasible_results)
            not_feasible_pathways.extend(not_feasible_results)

        return feasible_pathways, not_feasible_pathways
    
    def save_pathways(self, feasible_pathways, not_feasible_pathways, saved_file):

        # Save as pickle
        with open(f"{saved_file[:-4]}_Feasible.pkl", 'wb') as f:
            pickle.dump(feasible_pathways, f)
        with open(f"{saved_file[:-4]}_NotFeasible.pkl", 'wb') as f:
            pickle.dump(not_feasible_pathways, f)
        
        print(f"Saved both files")

    ################################     
    
def compute_reaction_fp(rxn_smiles, radius=2, use_chirality=True, use_features=True):
    """Compute reaction and product fingerprints for the query."""
    try:
        react_smi, prod_smi = rxn_smiles.split(">>")
        react_mol = Chem.MolFromSmiles(react_smi)
        prod_mol = Chem.MolFromSmiles(prod_smi)

        r_fp = AllChem.GetMorganFingerprint(react_mol, radius, 
                                            useChirality=use_chirality, 
                                            useFeatures=use_features)
        
        p_fp = AllChem.GetMorganFingerprint(prod_mol, radius, 
                                            useChirality=use_chirality, 
                                            useFeatures=use_features)
        
        rxn_fp = r_fp - p_fp
        return rxn_fp, p_fp
    
    except Exception as e:
        print(f"Error computing fingerprints for {rxn_smiles}: {e}")
        return None, None
    
def add_missing_components(rhea_id=None, precursors=None, target=None, 
                           missing_comp_file="../data/missing_rhea_components.csv",debug=False):
    
    missing_data_df = pd.read_csv(missing_comp_file)
    missing_data = missing_data_df[['id', 'missing_comp_left', 'missing_comp_right']]

    matching_row = missing_data[missing_data['id'] == rhea_id]

    # Check if matching_row is empty
    if matching_row.empty:
        if debug:
            print(f"Warning: No cofactor/cosubstrate data found for RHEA ID {rhea_id}")
        # Return a basic reaction without cofactors
        return '.'.join(precursors) + '>>' + target

    comp_left = matching_row['missing_comp_left'].iloc[0]    
    comp_right = matching_row['missing_comp_right'].iloc[0]

    # Build left side (from precursors)
    left_molecules = precursors
    if not pd.isna(comp_left):
        for comp in comp_left.split('.'):
            if comp not in precursors and comp != "nan":
                left_molecules.append(comp)
                
    # Build right side (from target)
    right_molecules = [target]
    if not pd.isna(comp_right):
        for comp in comp_right.split('.'):
            if comp != target and comp != "nan":
                    right_molecules.append(comp)
        
    # fully assembled reaction
    full_reaction = '.'.join(left_molecules) + '>>' + '.'.join(right_molecules)

    if debug:
        print(f"Adding missing reaction components: {full_reaction}")

    return full_reaction

def traverse_pathway_tree(node, path_steps=None):

    if path_steps is None:
        path_steps = []
    
    for reaction in node.reactions:
        reaction_info = {
            'source':reaction.source,
            'reaction_name': reaction.reaction_name,
            'parent_smiles': node.smiles,
            'precursor_smiles': [precursor.smiles for precursor in reaction.precursors],
            'precursors': reaction.precursors
            }
        
        path_steps.append(reaction_info)
        for precursor in reaction.precursors:
            traverse_pathway_tree(precursor, path_steps)

    return path_steps

####################### Check buyability functions ####################

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

    # Execute an SQL query to check if SMILES is in the specified table
    cursor.execute(f'SELECT 1 FROM {table_name} WHERE SMILES = ?', (smile,))
    result = cursor.fetchone() # Get first row from query result (or None if no rows match)
    return bool(result)
        
def check_availability(smile:str, database:sqlite3.Cursor) -> bool:
    
    result = check_in(smile, database, 'buyable')
    if not result:
        # try neutralizing it
        n_smi = neutralize_smiles(smile)
        return check_in(n_smi, database, 'buyable')
    return result

#######################################################################

####################### Visualization functions #######################

# For single-step planner output
def display_single_step_results(target:str, result_set:dict, buyables:sqlite3.Cursor = None, custom:sqlite3.Cursor = None):
    """Display retrosynthetic steps with buyability information"""

    print(f'There are a total of {len(result_set)} proposed single-step reactions')
        
    for item in result_set:
        print(f"{item[0]} {item[1]}")
        
        if item[0] == 'RHEA':
            print(f"https://www.rhea-db.org/rhea/{item[1][:-2] if '-' in item[1] else item[1]}")
            
        proposed_rxn = item[3] + ">>" + target
        print(proposed_rxn)
        display(draw_reaction(proposed_rxn))
        
        for prec in item[3].split('.'):
            is_buyable = check_availability(prec, buyables)
            is_custom = check_availability(prec, custom)
            if is_buyable: 
                print(f'*Available: {prec}')
            elif is_custom:
                print(f'*(Custom) Available: {prec}')
            else:
                print(f'*Not available: {prec}')
                
        print('\n')

# For multi-step planner output
def display_pathway_step(step, step_num, total_steps, root_smiles, is_unfeasible=False, buyables=None, custom=None):
    feas_suffix = " NOT FEASIBLE" if is_unfeasible else ""
    print(f"STEP {step_num}/{total_steps}{feas_suffix}")

    available_list = []
    prec = step['precursor_smiles']
    target = step['parent_smiles'] if step['parent_smiles'] else root_smiles
    _id = step['reaction_name']

    if step['source'] == 'RHEA':
        print(f"https://www.rhea-db.org/rhea/{int(_id)}")
        full_rxn = add_missing_components(rhea_id=_id, precursors=prec, target=target, debug=False)

    else:
        print(f"RetroBioCat - {_id}")
        full_rxn = '.'.join(prec) + ">>" + target

    react = full_rxn.split('>>')[0]
    for mol in react.split('.'):
        if check_availability(mol, buyables) or check_availability(mol, custom):
            available_list.append(mol)


    mapped_rxn = map_reaction(full_rxn, debug=False)
    img = draw_reaction_with_symbols(mapped_rxn, prec, target, available_list, mol_size=(250, 150))
    display(img)
    print(f"Reaction SMILES: {full_rxn}\n")
    
    return _id

def display_pathway(path_root, path_idx, unfeasible_steps=None, buyables=None, custom=None):
    
    print(f"PATH {path_idx} {'==='*10}")

    if unfeasible_steps:
        print(f"\nA total of {len(unfeasible_steps)} steps were marked as unfeasible")
        for unfeas_step in unfeasible_steps:
            print(f"  - Step {unfeas_step['step_number']}: {unfeas_step['reaction_name']} ({unfeas_step['feasibility']})")
        print()

    ids = []
    steps = traverse_pathway_tree(path_root)
    for i, step in enumerate(steps):
        step_num = i + 1
        is_unfeasible = False

        if unfeasible_steps:
            is_unfeasible = any(us['step_number'] == step_num for us in unfeasible_steps)

        _id = display_pathway_step(step, step_num, len(steps), path_root.smiles, is_unfeasible, buyables, custom)
        ids.append(_id)

    print(f"Reactions: {ids}\n")

def display_all_pathways(pathways, buyables, custom):

    print(f"There are a total of {len(pathways)} pathways")

    sorted_paths = sorted(pathways, key=lambda path: len(traverse_pathway_tree(node=path)))
    for idx, path in enumerate(sorted_paths):
        display_pathway(path, idx + 1, None, buyables, custom)

def display_feasible_pathways(feasible_pathways, buyables, custom):
        
    print(f"{len(feasible_pathways)} pathways are feasible")

    sorted_paths = sorted(feasible_pathways, key=lambda path: len(traverse_pathway_tree(node=path[0])))
    for idx, path in enumerate(sorted_paths):
        display_pathway(path[0], idx + 1, None, buyables, custom)

def display_unfeasible_pathways(not_feasible_pathways, buyables, custom):

    print(f"{len(not_feasible_pathways)} pathways are NOT feasible")

    sorted_unfeasible = sorted(not_feasible_pathways, key=lambda path: len(traverse_pathway_tree(node=path[0])))
    for idx, (path_root, unfeasible_steps) in enumerate(sorted_unfeasible):
        display_pathway(path_root, idx + 1, unfeasible_steps, buyables, custom)








        

        

            
                
                    



    
