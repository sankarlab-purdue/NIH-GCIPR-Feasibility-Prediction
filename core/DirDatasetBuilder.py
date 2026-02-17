import pandas as pd
import os
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import rdkit.RDLogger as RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from multiprocessing import Pool
from ReactChemChecker import *
import re

def get_crosslinks(swissprot_db, other_db):
    """
    Identify direction-specific crosslinks in Rhea database
    """

    # Load data
    swissprot = pd.read_csv(swissprot_db, delimiter='\t')
    other_db = pd.read_csv(other_db, delimiter='\t')

    # SWISSPROT
    direction_id = {}
    for _, row in swissprot.iterrows():
        master_id = row["MASTER_ID"] 
        direction = row["DIRECTION"]

        if master_id not in direction_id:
            direction_id[master_id] = []
        if direction not in direction_id[master_id]:
            direction_id[master_id].append(direction)

    #OTHER DB
    # Exclude KEGG, EC and Macie, as these do not assign directionality
    filtered_db = other_db[(other_db['DB'] != 'KEGG_REACTION') & (other_db['DB'] != 'EC')  & (other_db['DB'] != 'MACIE')]
    # Remove all entries that are UN -- we need info on directionality
    filtered_db = filtered_db[(filtered_db['DIRECTION'] != 'UN')]

    for _, row in filtered_db.iterrows():
        master_id = row["MASTER_ID"] 
        direction = row["DIRECTION"]

        if master_id not in direction_id:
            direction_id[master_id] = []
        if direction not in direction_id[master_id]:
            direction_id[master_id].append(direction)

    # Organize IDs based on directionalities
    lr_rxn = set()
    rl_rxn = set()
    bi_rxn = set()

    for key, value in direction_id.items():
        if len(value) == 1:
            if "LR" in value:
                lr_rxn.add(key)
            elif "RL" in value:
                rl_rxn.add(key)
            elif "BI" in value:
                bi_rxn.add(key)
        if len(value) == 2:
            if "UN" in value:
                if "LR" in value:
                    lr_rxn.add(key)
                elif "RL" in value:
                    rl_rxn.add(key)
            if "BI" in value:
                bi_rxn.add(key)
        if len(value) == 3:
            bi_rxn.add(key)

        if len(value) == 4:
            bi_rxn.add(key)

    # Now adding these to the directionality dataset...
    directionality_dataset = {'RHEA_ID': [], 'direction': [], 'feasible?': []}

    # first, I'll append the LR reactions (NORMAL)
    for item in lr_rxn:
        lr_id = item + 1
        rl_id = item + 2

        # lr item
        directionality_dataset['RHEA_ID'].append(lr_id)
        directionality_dataset['direction'].append("LR")
        directionality_dataset['feasible?'].append("Yes") 

        # rl item
        directionality_dataset['RHEA_ID'].append(rl_id)
        directionality_dataset['direction'].append("RL") 
        directionality_dataset['feasible?'].append("No") 

    # Then, I'll append the RL reactions (NORMAL)
    for item in rl_rxn:
        lr_id = item + 1
        rl_id = item + 2

        # lr item
        directionality_dataset['RHEA_ID'].append(lr_id)
        directionality_dataset['direction'].append("LR")
        directionality_dataset['feasible?'].append("No") 

        # rl item
        directionality_dataset['RHEA_ID'].append(rl_id)
        directionality_dataset['direction'].append("RL") 
        directionality_dataset['feasible?'].append("Yes") 

    # Finally, the BI reactions (BI)   
    for item in bi_rxn:
        lr_id = item + 1
        rl_id = item + 2

        # lr item
        directionality_dataset['RHEA_ID'].append(lr_id)
        directionality_dataset['direction'].append("LR")
        directionality_dataset['feasible?'].append("Yes") 

        # rl item
        directionality_dataset['RHEA_ID'].append(rl_id)
        directionality_dataset['direction'].append("RL") 
        directionality_dataset['feasible?'].append("Yes")

    return directionality_dataset

def clean_smiles(reaction_smiles):
    '''
    Remove protons and "*" stand-alone groups, as these are not 
    treated correctly when rules are extracted and applied

    Change [n*] to [*] to maintain consistency

    '''
    
    exclude_components = ["[H+]", "*", "[H]*[H]"]
    
    react, prod = reaction_smiles.split(">>")
    
    clean_react = []
    for item in react.split("."):
        if not item in exclude_components:
            clean_react.append(item)
    
    clean_prod = []
    for item in prod.split("."):
        if not item in exclude_components:
            clean_prod.append(item)
        
    final_clean_react = ".".join(clean_react)
    final_clean_prod = ".".join(clean_prod)
        
    if '[1*]' in final_clean_react:
        final_clean_react = re.sub(r'\[\d+\*\]', '[*]', ".".join(clean_react))
    if '[1*]' in final_clean_prod:
        final_clean_prod = re.sub(r'\[\d+\*\]', '[*]', ".".join(clean_prod))
    
    return final_clean_react+">>"+final_clean_prod

def add_smiles(directionality_dataset, rhea_smiles_db):
    """
    Add cleaned smiles to directionality dataset
    """

    # Load data
    rhea_smiles = pd.read_csv(rhea_smiles_db, delimiter='\t', names=['RHEA_ID', 'rxn_smiles'])

    directionality_dataset["rxn_smiles"] = []

    for item in directionality_dataset["RHEA_ID"]:
        matching_row = rhea_smiles[rhea_smiles["RHEA_ID"] == item]

        if matching_row.empty:
            directionality_dataset["rxn_smiles"].append("n/a")
            continue

        rxn = matching_row["rxn_smiles"].iloc[0]
        rct = rxn.split(">>")[0]
        if len(rct.split(".")) >= 8: 
            directionality_dataset["rxn_smiles"].append("n/a")
        else:
            cleaned_rxn = clean_smiles(rxn)
            directionality_dataset["rxn_smiles"].append(cleaned_rxn)

    # Remove non valid smiles 
    directionality_dataset = pd.DataFrame.from_dict(directionality_dataset)
    directionality_dataset = directionality_dataset.sort_values(by=['RHEA_ID'], ascending=True)
    directionality_dataset = directionality_dataset[~directionality_dataset['rxn_smiles'].str.contains('n/a')]

    return directionality_dataset

def remove_transport_reactions(directionality_dataset, rhea_transport_ids):
    """
    Remove entries that represent transport reactions
    """

    # Load data
    transport_data = pd.read_csv(rhea_transport_ids, delimiter='\t')

    transport_id = []
    for item in transport_data["Reaction identifier"]:
        _id = int(item.split(':')[1])
        transport_id.append(_id)

    print(f"   -Total number of UN Transport IDs: {len(transport_id)}")

    transport_id_lr_rl = []
    for item in transport_id:
        transport_id_lr_rl.append(str(item + 1))
        transport_id_lr_rl.append(str(item + 2))

    no_transport_filtered_dataset = directionality_dataset[~directionality_dataset['RHEA_ID'].astype(str).str.contains('|'.join(transport_id_lr_rl))]

    return no_transport_filtered_dataset

def complement_entries(directionality_dataset):
    """
    Add columns for canonicalized reaction smiles and product smiles.
    Will be important for fingerprinting later on.
    """

    directionality_dataset = directionality_dataset.copy()

    can_rxn_smiles = []
    for item in directionality_dataset['rxn_smiles']:
        can_item = canonicalize_reaction(item)
        can_rxn_smiles.append(can_item)

    # Add it to the dataframe
    directionality_dataset["can_rxn_smiles"] = can_rxn_smiles

    # Column for products
    rxn_products = []
    for item in directionality_dataset['can_rxn_smiles']:
        prod = item.split(">>")[1]
        rxn_products.append(prod)

    directionality_dataset["rxn_products"] = rxn_products

    print(f"   -Example: \n{directionality_dataset[:1]}")

    return directionality_dataset

### Parallel analysis function for template checker ###

def process_single_reaction(row_data):
    """ Reaction processing for each worker during parallelization """
    rhea_id, can_rxn_smiles, feas = row_data

    try:
        # Get IDs with RXNMapper issues
        if not map_reaction(can_rxn_smiles):
            return ('exception', {'rhea_id': rhea_id, 'error': 'RXNMapper issue - Template cannot be extracted', 'smiles': can_rxn_smiles})

        # Extract reaction template
        temp = template_extractor(can_rxn_smiles)

        if temp:
            result_data = {
                "_id": rhea_id,
                "rxn_smiles": can_rxn_smiles,
                "template": temp,
                "feas": feas
            }
            return ('success', result_data)

        else:
            return ('exception', {'rhea_id': rhea_id, 'error': 'temp as None', 'smiles': can_rxn_smiles})

    except Exception as e:
        return ('exception', {'rhea_id': rhea_id, 'error': str(e), 'smiles': can_rxn_smiles})
    
        
def template_checker(data, n_processes=None):
    """
    Checks if templates can be extracted from the reaction SMILES
    """
    
    # Get number of processes
    if n_processes is None:
        try:
            n_processes = len(os.sched_getaffinity(0))
        except:
            n_processes = 1

    input_data = [(row["RHEA_ID"], row["can_rxn_smiles"], row["feasible?"]) for _, row in data.iterrows()]
    
    # Process in parallel 
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(tqdm(
            executor.map(process_single_reaction, input_data),
            total=len(input_data),
            desc="Processing reactions"
        ))

    template_dict = {"_id": [], "rxn_smiles": [], "template": [], "feas": []}
    exceptions = []

    for status, result_data in results:
        if status == 'success':
            template_dict["_id"].append(result_data["_id"])
            template_dict["rxn_smiles"].append(result_data["rxn_smiles"])
            template_dict["template"].append(result_data["template"])
            template_dict["feas"].append(result_data["feas"])

        elif status == 'exception':
            exceptions.append(result_data)

    base_template_dataset = pd.DataFrame.from_dict(template_dict)
    template_dataset = base_template_dataset[base_template_dataset['template'].notna()].copy()
    ids_to_keep = template_dataset['_id'].tolist()
    filtered_feasibility_data = data[data['RHEA_ID'].isin(ids_to_keep)].copy()

    print(f"    -Successfully processed: {len(filtered_feasibility_data)} reactions")
    print(f"    -Exceptions: {len(exceptions)} reactions")

    return filtered_feasibility_data, template_dataset, exceptions

######################################################

def get_template_count(template_dataset):
    """
    Check how many distinct templates are in the dataset
    """
    # Group by template
    grouped = template_dataset.groupby('template')

    results = []

    for template, group in grouped:
        feasible_ids = group[group['feas'] == 'Yes']['_id'].tolist()
        not_feasible_ids = group[group['feas'] == 'No']['_id'].tolist()

        feas_rhea_ids_str = ';'.join(map(str, feasible_ids))
        not_feas_rhea_ids_str = ';'.join(map(str, not_feasible_ids))

        # Count total entries
        total_count = len(group)

        # Count feasible and not feasible
        feasible_count = len(feasible_ids)
        not_feasible_count = len(not_feasible_ids)

        results.append({
            'List of feasible RHEA IDs': feas_rhea_ids_str,
            'List of not feasible RHEA IDs': not_feas_rhea_ids_str,
            'Template': template,
            'Count': total_count,
            'Number of feasible': feasible_count,
            'Number of not feasible': not_feasible_count
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Sort by count (descending) to see most common templates first
    template_count = results_df.sort_values('Count', ascending=False)
    
    return template_count

### Parallel analysis function for template application ###

def process_single_row(args, debug_flag=True):

        row_data, debug_flag = args
        idx, row = row_data
        row_debug = ""
        result = "no_match"
        rdchiral_increment = 0
        achiral_increment = 0
        rdkit_increment = 0

        # Extract the reaction SMILES from the row
        rxn_smiles = row['can_rxn_smiles']

        if debug_flag:
            row_debug += f"Processing reaction: {rxn_smiles}\n"        

        try: 
            # Run the template extractor/applicator function    
            template = template_extractor(rxn_smiles)
            
            generated_full_rxn, match_info = reaction_chemistry_checker(template, rxn_smiles)
            
            if match_info == 'rdchiral':
                rdchiral_increment += 1
            elif match_info == 'achiral':
                achiral_increment += 1
            elif match_info == 'rdkit':
                rdkit_increment += 1

            # If reaction_chemistry_checker returned something, it's a match
            if generated_full_rxn:
                result = "match"
                row_debug += f"Template model applied\n"
            else:
                row_debug += f"Match not found\n"
        except Exception as e:
            result = f"error: {str(e)}"
            row_debug += f"Exception encountered: {str(e)}\n"

        return {
            'idx': idx,
            'result': result,
            'debug_info': row_debug,
            'rdchiral_increment': rdchiral_increment,
            'achiral_increment': achiral_increment,
            'rdkit_increment': rdkit_increment
        }
    

def reaction_checker_validation(feasibility_data, n_processes=None, debug=False):
    """
    Reaction Chemistry Checker validation process
    """

    # Get number of processes
    if n_processes is None:
        try:
            n_processes = len(os.sched_getaffinity(0))
        except:
            n_processes = 1

    print(f"Using {n_processes} processes for parallel processing")

    # Prepare arguments for multiprocessing
    row_args = [((idx, row), debug) for idx, row in feasibility_data.iterrows()]

    # Process rows in parallel
    print("Processing reactions in parallel...")
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_row, row_args), 
            total=len(row_args),
            desc="Processing reactions"
        ))

    # Collect results
    validation_results = [''] * len(feasibility_data)
    debug_infos = [''] * len(feasibility_data)
    total_rdchiral = 0
    total_achiral = 0
    total_rdkit = 0

    # Sort results by original index to maintain order
    for result in results:
        idx = result['idx']
        # Find position in feasibility_data
        pos = list(feasibility_data.index).index(idx)
        validation_results[pos] = result['result']
        debug_infos[pos] = result['debug_info']
        total_rdchiral += result['rdchiral_increment']
        total_achiral += result['achiral_increment']
        total_rdkit += result['rdkit_increment']

    result_df = feasibility_data.copy()
    result_df['validation_result'] = validation_results
    result_df['debug_info'] = debug_infos

    total = len(result_df)
    matches = result_df['validation_result'].apply(lambda x: x == "match").sum()
    accuracy = (matches / total) * 100

    print(f"    -Processed {total} reactions. Accuracy: {accuracy:.2f}%")

    result_df_no_match = result_df[result_df['validation_result'] != "match"]
    result_df_match = result_df[result_df['validation_result'] == "match"]

    result_df_match_only = result_df_match.drop(['validation_result', 'debug_info'], axis=1)
    print(f"    -Total remaining data: {len(result_df_match_only)}")
    
    print(f"RDChiral count = {total_rdchiral}")
    print(f"Achiral count = {total_achiral}")
    print(f"RDKit count = {total_rdkit}")

    return result_df_match_only, result_df_no_match

################################################################

def remove_parent_ids(directionality_dataset, rhea_ids_db, rhea_parent_ids):
    """
    Remove entries that are general representations of specific reactions in Rhea
    """

    # Load data
    full_rhea_ids = pd.read_csv(rhea_ids_db, delimiter='\t')
    rhea_relationships = pd.read_csv(rhea_parent_ids, delimiter='\t')

    parents = rhea_relationships["TO_REACTION_ID"].tolist()
    print(f"   -Number of (Non-Unique) Parent IDs: {len(parents)}")
    parents_set = set(parents)
    print(f"   -Number of Unique Parent IDs: {len(parents_set)}")

    feasibility_dataset_copy = directionality_dataset.copy()

    full_rhea_ids_lr = full_rhea_ids["RHEA_ID_LR"].tolist()
    full_rhea_ids_rl = full_rhea_ids["RHEA_ID_RL"].tolist()
    feasibility_ids = []
    for item in feasibility_dataset_copy["RHEA_ID"].tolist():
        item = int(item)
        if item in full_rhea_ids_lr:
            feasibility_ids.append(item-1)
        elif item in full_rhea_ids_rl:
            feasibility_ids.append(item-2)

        else:
            print("error")

    unique_feasibility_ids = list(set(feasibility_ids))
    # How many undefined IDs we have in feasibility dataset?
    print(f"   -Total UN IDs in the Feasibility Dataset: {len(unique_feasibility_ids)}")

    overlap = [] 
    for item in unique_feasibility_ids:
        if item in parents_set:
            overlap.append(item)

    print(f"   -Number of parent IDs in the Feasibility Dataset: {len(overlap)} - {overlap[:5]}")

    print("   -Original length:", len(feasibility_dataset_copy))
    for item in overlap:
        item_lr = item + 1
        item_rl = item + 2

        # Save the filtered results back to feasibility_data
        feasibility_dataset_copy = feasibility_dataset_copy[
            (feasibility_dataset_copy["RHEA_ID"] != item_lr) & 
            (feasibility_dataset_copy["RHEA_ID"] != item_rl)
        ]

    print("   -Final length:", len(feasibility_dataset_copy))

    return feasibility_dataset_copy

def analyze_template_duplicates(template_dataset):
    """
    Analyze template duplicates and their feasibility
    """
    # Group by template
    grouped = template_dataset.groupby('template')
    
    results = []
    
    for template, group in grouped:
        feasible_ids = group[group['feas'] == 'Yes']['_id'].tolist()
        not_feasible_ids = group[group['feas'] == 'No']['_id'].tolist()
        
        feas_rhea_ids_str = ';'.join(map(str, feasible_ids))
        not_feas_rhea_ids_str = ';'.join(map(str, not_feasible_ids))
        
        # Count total entries
        total_count = len(group)
        
        # Count feasible and not feasible
        feasible_count = len(feasible_ids)
        not_feasible_count = len(not_feasible_ids)
        
        results.append({
            'List of feasible RHEA IDs': feas_rhea_ids_str,
            'List of not feasible RHEA IDs': not_feas_rhea_ids_str,
            'Template': template,
            'Count': total_count,
            'Number of feasible': feasible_count,
            'Number of not feasible': not_feasible_count
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by count (descending) to see most common templates first
    results_df = results_df.sort_values('Count', ascending=False)
    
    # Save to CSV
    results_df.to_csv('validation_output/template_analysis_results.csv', index=False)
    
    # Print summary statistics
    print(f"Total unique templates: {len(results_df)}")
    print(f"Templates with duplicates (count > 1): {len(results_df[results_df['Count'] > 1])}")
    print(f"Most common template appears {results_df['Count'].max()} times")
    
    return results_df

def remove_unique_temp(directionality_dataset, template_dataset):
    """
    Remove entries that have unique templates for cross-validation
    """

    template_counts = template_dataset['template'].value_counts()
    template_dataset['is_unique_template'] = template_dataset['template'].map(lambda x: template_counts[x] == 1)

    # keep entries where template is not unique
    filtered_dataset = template_dataset[(~template_dataset['is_unique_template'])].copy()
    filtered_dataset = filtered_dataset.drop('is_unique_template', axis=1)

    ids_to_keep = filtered_dataset['_id'].tolist()
    filtered_directionality_data = directionality_dataset[directionality_dataset['RHEA_ID'].isin(ids_to_keep)].copy()

    print(f"   -Non-unique templates (all kept):{len(filtered_directionality_data)}")

    return filtered_directionality_data

def precompute_fingerprints(file_name, df, smiles_col='can_rxn_smiles', radius=2, use_chirality=True, use_features=True):
    """
    Precompute Morgan fingerprints (reactants, products, and their difference)
    for each row in the DataFrame, storing them in new columns:
        df['react_fp'], df['prod_fp'], df['rxn_fp'].
    """
    react_fps = []
    prod_fps = []
    rxn_fps = []
    
    for i, row in df.iterrows():
        rxn_smi = row[smiles_col]
        try:
            react_smis, prod_smis = rxn_smi.split('>>')
            
            # Convert entire reactants/products side to a single Mol
            # (If multiple reactants are separated by '.', you can join them,
            # or parse them individually. For simplicity, let's parse them as one string.)
            react_mol = Chem.MolFromSmiles(react_smis)
            prod_mol  = Chem.MolFromSmiles(prod_smis)
            
            if react_mol is None or prod_mol is None:
                react_fps.append(None)
                prod_fps.append(None)
                rxn_fps.append(None)
                continue
            
            # Generate Morgan FPs
            r_fp = AllChem.GetMorganFingerprint(react_mol, radius,
                                                useChirality=use_chirality,
                                                useFeatures=use_features)
            p_fp = AllChem.GetMorganFingerprint(prod_mol, radius,
                                                useChirality=use_chirality,
                                                useFeatures=use_features)
            
            # Difference fingerprint
            rxn_fp = r_fp - p_fp
            
            react_fps.append(r_fp)
            prod_fps.append(p_fp)
            rxn_fps.append(rxn_fp)
            
        except:
            # If there's any parsing issue
            react_fps.append(None)
            prod_fps.append(None)
            rxn_fps.append(None)
    
    df['react_fp'] = react_fps
    df['prod_fp'] = prod_fps
    df['rxn_fp'] = rxn_fps

    output_pickle = f"../data/{file_name}.pkl"
    df.to_pickle(output_pickle)
    print(f"Done! Precomputed fingerprints stored in {output_pickle}")

    #return df