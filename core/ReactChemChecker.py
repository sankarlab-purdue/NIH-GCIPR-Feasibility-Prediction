from template_extractor import extract_from_reaction_fwd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun
import itertools
from rxnmapper import RXNMapper
from transformers import logging
logging.set_verbosity_error()


# Initialize model for reaction atom mapping
rxn_mapper = RXNMapper()
# Suppress transformer warnings
logging.set_verbosity_error()      

MATCH_EXACT = None
MATCH_ACHIRAL_RDCHIRAL = 'achiral-rdchiral'
MATCH_ACHIRAL_RDKIT = 'achiral-rdkit'
MATCH_RDKIT = 'rdkit'

def map_reaction(rxn_smiles, debug=False):
    """Add atom mapping to reaction using RXNMapper."""
    try:
        mapped_results = rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])
        return mapped_results[0]['mapped_rxn']
    except Exception as e:
        if debug:
            print(f"Couldn't atom map the reaction {rxn_smiles} - {e}")
        return None

def canonicalize_smiles(smiles):
    """Canonicalize SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else smiles
    except:
        return smiles    

def canonicalize_reaction(rxn_smiles):
    """Canonicalize reaction SMILES."""
    try:
        reactants, products = rxn_smiles.split('>>')

        # Canonicalize and sort each side
        canon_reactants = '.'.join(sorted([
            canonicalize_smiles(smi) for smi in reactants.split('.')
        ]))
        canon_products = '.'.join(sorted([
            canonicalize_smiles(smi) for smi in products.split('.')
        ]))

        return f"{canon_reactants}>>{canon_products}"
    except:
        return rxn_smiles
        
def draw_reaction(rxn_smiles, is_smarts=False, img_size=(500, 300)):
    """Draw reaction image."""
    try:
        if is_smarts:
            rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles)
        else:
            rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        return Draw.ReactionToImage(rxn, subImgSize=img_size)
    except Exception as e:
        print(f"Could not draw reaction: {e}")
        return None
    
def draw_molecule(mol_smiles, img_size=(200, 200)):
    """Draw molecule image."""
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            print(f"Could not parse molecule: {mol_smiles}")
            return None
        return Draw.MolToImage(mol, size=img_size)

    except Exception as e:
        print(f"Could not draw molecule: {e}")
        return None

def parse_reaction_smiles(rxn_smiles):
    """Parse reaction SMILES into reactants and products"""
    
    reactants_smiles, products_smiles = rxn_smiles.split('>>')

    return {
        'reactants': reactants_smiles,
        'products': products_smiles
    }

def template_extractor(reaction_smiles, debug=False):
    """Extract forward template using RDChiral"""
    
    try:
        # Canonicalization (Standard practice)
        reaction_smiles = canonicalize_reaction(reaction_smiles)
        
        if debug:
            print(f"Reaction for Template Extraction: {reaction_smiles}")
            display(draw_reaction(reaction_smiles, is_smarts=False))
            
        # Map reaction smiles for template extration
        mapped_reaction_smiles = map_reaction(reaction_smiles)
        
        parsed_smiles = parse_reaction_smiles(mapped_reaction_smiles)
        temp = extract_from_reaction_fwd(parsed_smiles)
        if temp:
            if debug:
                print(f"Forward template: {temp}")
                display(draw_reaction(temp, is_smarts=True))

            return temp
        else:
            return None
    
    except Exception as e:
        if debug:
            print(f"Error extracting template: {e} - {reaction_smiles}")
        return None
        
def reaction_chemistry_checker(temp, reaction_smiles, debug=False):
    """Check if template application produces expected products"""

    # Canonicalize reaction
    reaction_smiles = canonicalize_reaction(reaction_smiles)
    if debug:
        print(f"\nReaction for Template Application: {reaction_smiles}")
        display(draw_reaction(reaction_smiles, is_smarts=False))

    # Split reaction into 'reactants for application' and 'products expected to be generated'
    reactants, products_expected = reaction_smiles.split(">>")
    if debug:
        print(f"Reactants that will undergo template application: {reactants}")

    # Remove any products that are also reactants (unchanged molecules)
    filtered_products_list = [p for p in products_expected.split('.') if p not in reactants.split('.')]
    filtered_products_expected = '.'.join(filtered_products_list)

    try:
        return _use_rdchiral(temp, reactants, filtered_products_expected, debug)
    except Exception as e:
        if debug:
            print(f"RDChiral template application failed: {e}")
        return _use_rdkit(temp, reactants, filtered_products_expected, debug)

def _use_rdchiral(temp, reactants, products_expected, debug=False):
        """Apply reaction template using rdchiral"""

        # Prepare template
        template_split = temp.split(">>")
        mod_temp = f"({template_split[0][1:-1].replace(').(', '.')})>>{template_split[1][1:-1].replace(').(', '.')}"
    
        # Run rdchiral
        rxn = rdchiralReaction(mod_temp)
        rct = rdchiralReactants(reactants)
        outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)

        if debug:
            for idx, item in enumerate(outcomes):
                print(f'Outcome {idx} SMILES: {item}')

        if outcomes:
            for item in outcomes:
                if all(prd in item.split('.') for prd in products_expected.split('.')):
                    if debug:
                        print(f'Correct outcome identified: {item}')     
                        display(Chem.MolFromSmiles(item))
                    return True, 'rdchiral'

            # There are outcomes, but no match
            # Check if there is an achiral match
            return achiral_match(products_expected, outcomes, debug=debug)
        return False, 'rdchiral'

def _use_rdkit(temp, reactants, products_expected, debug=False):
    """Apply reaction template using rdkit"""

    try:
        # Convert SMILES to molecules
        reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactants.split('.')]
        
        if None in reactant_mols:
            print("Could not parse reactant SMILES")
            return False, 'rdkit'
        
        for mol in reactant_mols:
            Chem.SanitizeMol(mol)
        
        # Create reaction from template
        rxn = AllChem.ReactionFromSmarts(temp)
        
        # Validate reaction
        if rxn.Validate()[1] != 0:
            print('Could not validate reaction template')
            return False, 'rdkit'
        if debug:
            print(f"Template requires {rxn.GetNumReactantTemplates()} reactants")
            print(f"Provided {len(reactant_mols)} reactants")
        
        # If there are 4 reactants, but the template requires only 2, it will test every permutation of 2
        combinations = itertools.permutations(reactant_mols, rxn.GetNumReactantTemplates())
        
        unique_products = set()
        for combination in combinations:
            if debug:
                print(f"Trying combination: {[Chem.MolToSmiles(mol) for mol in combination]}")
                
            outcomes = rxn.RunReactants(list(combination))
            if not outcomes:
                if debug:
                    print(f"   No outcomes generated for this combination!!")
                continue
                
            for j, outcome in enumerate(outcomes):
                if debug:
                    print(f"Outcome {j+1}/{len(outcomes)}:")

                try:    
                    product_smiles = []
                    for product in outcome:
                        Chem.SanitizeMol(product)
                        product.UpdatePropertyCache()
                        product_smiles.append(Chem.MolToSmiles(product, isomericSmiles=True))

                    canonical_products = '.'.join(sorted(canonicalize_smiles(s) for s in product_smiles))
                    unique_products.add(canonical_products)
                    if debug:
                        print(f"   Products: {canonical_products}")

                except Exception as e:
                    if debug:
                        print(f"   Warning: Could not process product: {e}")
                        
        if unique_products:
            for item in unique_products:
                if all(prd in item.split('.') for prd in products_expected.split('.')):
                    if debug:
                        print(f'Correct outcome identified: {item}')
                        display(draw_molecule(item))
                    return True, 'rdkit'
                
            # There are outcomes, but no match
            # Check if there is an achiral match
            return achiral_match(products_expected, unique_products, debug=debug)
        return False, 'rdkit'
        
    except Exception as e:
        if debug:
            print(f"Error applying template using RDKit: {e}")
        return False, 'rdkit'
    
def achiral_match(products_expected, outcomes, debug=False):
    
    if debug:
        print('Performing achiral check')
    
    achiral_expected = [achiral(prod) for prod in products_expected.split('.')]
    achiral_outcomes = []
    for item in outcomes:
        achiral_item = [achiral(mol) for mol in item.split('.')]
        achiral_outcomes.append(achiral_item)
        
    for item in achiral_outcomes:
        if all(prds in item for prds in achiral_expected):
            if debug:
                print(f'Correct ACHIRAL outcome identified - Possibly stereochem issue? {item}')     
                display(Chem.MolFromSmiles('.'.join(item)))
                
            return True, 'achiral'
    return False, 'achiral'
    

def achiral(smiles):
    '''
    Get achiral version of a molecule
    '''
    
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)