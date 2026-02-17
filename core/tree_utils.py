from nodes import *
from template_extractor import *
from typing import List, Tuple
from itertools import product
from typing import List
from utils import Availability, Retrosim
import sqlite3
import pickle
from nodes import ChemNode
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from IPython.display import Image
from PIL import Image, ImageDraw, ImageFont


def path_explorer(root: ChemNode):
    """
    Prints the path given a subtree

    :param root: The root of the solution subtree.
    """

    # Initialize stack of chemicals that branch out
    stack = []
    stack.append(root)
    count = 0
    
    # Easier to find specific rxns
    saved_rxn = []
    
    while stack:
        print(f"BRANCH {count}:\n-----------------")
        count += 1
        root = stack.pop()
        print(f'Chem: {root.smiles}')

        while root.reactions:
            if len(root.reactions[0].precursors) == 0:
                break

            print(f'Reaction: {root.reactions[0].reaction_name}, smarts:{root.reactions[0].smarts}')
            saved_rxn.append(root.reactions[0].reaction_name)

            if len(root.reactions[0].precursors) > 1:
                print(f"SPLIT: Chem1: {root.reactions[0].precursors[0].smiles}, Chem2: {root.reactions[0].precursors[1].smiles}\n")
                
                chems = []
                chems.append(root.reactions[0].precursors[0].smiles)
                chems.append(root.reactions[0].precursors[1].smiles)
                print(f"{chems}\n")
                # Add the second precursor to branch out
                stack.append(root.reactions[0].precursors[1])
            else:
                print(f'Chem: {root.reactions[0].precursors[0].smiles}')

                chems = []
                chems.append(root.reactions[0].precursors[0].smiles)
                
            # Establish first precursor as the new root to continue traversing the tree 
            root = root.reactions[0].precursors[0]
        status = root.get_availability() 
        if status == Availability.BUYABLE:
            print("BUYABLE\n")
        else: # status == Availability.NONE:
            print("Huh\n")
            
    print(f"Reactions: {saved_rxn}")

def path_explorer_str(root: ChemNode):
    """
    Returns the path given a subtree as a string

    :param root: The root of the solution subtree.
    """

    result = []

    # Initialize stack of chemicals that branch out
    stack = []
    stack.append(root)
    count = 0
    while stack:
        result.append(f"BRANCH {count}:\n-----------------")
        count += 1
        root = stack.pop()
        result.append(f'Chem: {root.smiles}')

        while root.reactions:
            result.append(f'Reaction: {root.reactions[0].reaction_name}, smarts:{root.reactions[0].smarts}')

            if len(root.reactions[0].precursors) > 1:
                result.append(f"SPLIT: Chem1: {root.reactions[0].precursors[0].smiles}, Chem2: {root.reactions[0].precursors[1].smiles}\n")
                # Add the second precursor to branch out
                stack.append(root.reactions[0].precursors[1])
            else:
                result.append(f'Chem: {root.reactions[0].precursors[0].smiles}')
            # Establish first precursor as the new root to continue traversing the tree 
            root = root.reactions[0].precursors[0]
        status = root.get_availability() 
        if status == Availability.SYNTHESIZABLE:
            result.append("SYNTHESIZABLE\n")
        elif status == Availability.BUYABLE:
            result.append("BUYABLE\n")
        elif status == Availability.NONE:
            result.append("Huh\n")

    return '\n'.join(result)

def generate_paths(root: ChemNode) -> List[ChemNode]:
    """
    Recursively generate all solution paths from a given ChemNode.
    Each ChemNode in the returned tree will have exactly one child reaction,
    and each reaction can have multiple precursor ChemNodes.

    :param root: The root of the tree
    """
    # Base case: if the node is a leaf solution (flagged and no reactions)
    if root.solution and not root.reactions:
        leaf = root.copy()
        # leaf.solution = False  # Optionally mark as processed
        return [leaf]
    
    all_paths = []
    # Iterate over each reaction option (OR branch) at the current node.
    for reaction in root.reactions:
        if not reaction.is_solved:
            continue
        precursor_solution_lists = []
        valid_reaction = True
        # For an AND reaction, all precursors must provide at least one solution.
        for precursor in reaction.precursors:
            # If the precursor isn't flagged or doesn't lead to a full pathway, skip this reaction.
            if not precursor.solution:
                valid_reaction = False
                break
            sub_paths = generate_paths(precursor)
            if not sub_paths:
                valid_reaction = False
                break
            precursor_solution_lists.append(sub_paths)
        
        # Only continue if every precursor returned at least one solution path.
        if not valid_reaction:
            continue
        
        # Combine precursor solutions: each combination yields a complete branch. using cartesian product
        for combo in product(*precursor_solution_lists):
            node_copy = root.copy()
            reaction_copy = reaction.copy()
            # Attach each precursor solution from the combination as a reactant to the reaction.
            for precursor_path in combo:
                reaction_copy.add_reactant(precursor_path)
            # Now the ChemNode gets a single child reaction (the copied reaction)
            node_copy.reactions = [reaction_copy]
            all_paths.append(node_copy)
    
    # If no reaction yielded a full pathway, mark this node as not part of a solution.
    if not all_paths:
        # node.solution = False
        pass
    return all_paths


def save_tree(save:str, file_names:List[str], root:ChemNode):
    """
    Save the tree to a file.

    :param save: The file to save the tree to.
    :param file_names: List of file locations where the databases are stored.
    :param root: The root of the tree.
    """
    ChemNode.custom = None
    ChemNode.buyables = None
    ChemNode.excluded = None
    ChemNode.analyzer = None
    with open(save, 'wb') as f:
        pickle.dump(file_names, f)
        pickle.dump(root, f)
        

def load_tree(save:str, buyable:sqlite3.Cursor = None, custom:sqlite3.Cursor = None,
              excluded:sqlite3.Cursor = None, analyzer:Retrosim = None) -> Tuple[ChemNode, List[str]]:
    """
    Load the tree from a file.

    :param save: The file to load the tree from.
    :param buyable: The buyable database if needed, otherwise it will be loaded from the file.
    :param custom: The virtual database if needed, otherwise it will be loaded from the file.
    :param excluded: The excluded database if needed, otherwise it will be loaded from the file.
    :param analyzer: The retrosim object if needed, otherwise it will be loaded from the file.

    :return: The root of the tree and the list of file names.
    """
    with open(save, 'rb') as f:
        files = pickle.load(f)
        root = pickle.load(f)

    if buyable is None:
        buyable = sqlite3.connect(f'file:{files[0]}', uri=True).cursor()
    if custom is None:
        custom = sqlite3.connect(f'file:{files[1]}', uri=True).cursor()
    if excluded is None:
        excluded = sqlite3.connect(f'file:{files[2]}', uri=True).cursor()

    if analyzer is None:
        analyzer = Retrosim(files[3], files[4], files[5])
        
    ChemNode.custom = custom
    ChemNode.buyables = buyable
    ChemNode.excluded = excluded
    ChemNode.analyzer = analyzer
    return root, files

def convert_rxn_smarts_to_mols(rxn_smarts):
    """Parse a reaction SMARTS string and convert it into separate lists of reactant and product RDKit molecule objects"""

    reactants_smiles, products_smiles = rxn_smarts.split('>>')
    reactants = mols_from_smiles_list(replace_deuterated(reactants_smiles).split('.'))
    products = mols_from_smiles_list(replace_deuterated(products_smiles).split('.'))
    
    return reactants, products
                                     

def map_changed_tags_to_indices(changed_atom_tags, mols):
    """
    Convert atom map tags to molecule/atom indices for visualization.
    
    Args:
        changed_atom_tags: List of atom map number strings
        mols: List of RDKit molecules
        
    Returns:
        Dict mapping mol_idx to list of atom indices that have changed
        e.g., {0: [2, 5], 1: [0, 3]}
    """
    changed_indices = {}
    changed_tags_set = set(changed_atom_tags) 
    
    for mol_idx, mol in enumerate(mols):
        matching_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.HasProp('molAtomMapNumber') 
            and atom.GetProp('molAtomMapNumber') in changed_tags_set
        ]
        
        if matching_atoms:
            changed_indices[mol_idx] = matching_atoms
    
    return changed_indices
    
def should_exclude_molecule(mol):
    """Check if molecule should be excluded from visualization (e.g., H+) """
    if mol is None or mol.GetNumAtoms() != 1:
        return mol is None
    
    # Exclude [H+]
    atom = mol.GetAtomWithIdx(0)
    return atom.GetSymbol() == 'H' and atom.GetFormalCharge() == 1    

def draw_individual_molecule(mol, highlight_atoms, atom_colors, mol_size):
    """
    Draw a single molecule with size-optimized spacing.
    
    Args:
        mol: RDKit Mol object to draw
        highlight_atoms: List of atom indices to highlight
        atom_colors: Dict mapping atom indices to RGB color tuples
        mol_size: Target (width, height) tuple for output image
        
    Returns:
        PIL Image of the molecule, or None if mol is None
    """
    
    if mol is None:
        return None
    
    # Scale factor based on molecule complexity
    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 3:
        scale_factor = 0.4 # small mol, zoom in
    elif num_atoms <= 10:
        scale_factor = 0.6 # medium mol, moderate zoom
    else:
        scale_factor = 0.9 # large mol, minimal zoom
    
    # Draw at larger size then scale down (effectively zooms in on molecule)
    draw_options = Draw.DrawingOptions()
    draw_options.padding = 0.05
    
    temp_size = (int(mol_size[0] / scale_factor), int(mol_size[1] / scale_factor))
    img = Draw.MolToImage(
        mol, size=temp_size,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        options=draw_options
    )
    
    return img.resize(mol_size, Image.LANCZOS)

def draw_reaction_with_symbols(rxn_smarts, precursors, target, buyables,
                                change_color=(1.0, 0.7, 0.7),
                                mol_size=(300, 300),
                                verbose=False):
    """
    Draw a chemical reaction with comprehensive highlighting of changed atoms and buyable labels.
    
    Args:
        rxn_smarts: Atom-mapped reaction SMARTS string
        precursors: List of precursor SMILES strings
        target: Target molecule SMILES (unused but kept for API compatibility)
        buyables: Set/list of buyable molecule SMILES
        change_color: RGB tuple for highlighting changed atoms (default: light red)
        mol_size: Size tuple for individual molecule images
        verbose: Enable verbose output from change detection
        
    Returns:
        PIL Image showing the complete reaction with annotations
    """
    
    # Convert reaction SMARTS to molecule lists 
    reactants_orig, products_orig = convert_rxn_smarts_to_mols(rxn_smarts)
    
    # Detecting change
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants_orig, products_orig)
    
    if verbose:
        print(f"Found {len(changed_atom_tags)} changed atoms: {changed_atom_tags}")
    
    # Map changed tags to indices for both reactants and products
    changed_reactants = map_changed_tags_to_indices(changed_atom_tags, reactants_orig)
    changed_products = map_changed_tags_to_indices(changed_atom_tags, products_orig)
    
    # Process molecules for visualization
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    reactant_data, buyable_indices = _process_reactants(rxn, changed_reactants, precursors, buyables, change_color)
    product_data, target_index = _process_products(rxn, changed_products, target, change_color)
    
    # Render individual molecules
    reactant_images = [draw_individual_molecule(mol, hl, col, mol_size) 
                       for mol, hl, col, _ in reactant_data]
    product_images = [draw_individual_molecule(mol, hl, col, mol_size) 
                      for mol, hl, col in product_data]
    
    reactant_images = [img for img in reactant_images if img]
    product_images = [img for img in product_images if img]
    
    if not reactant_images or not product_images:
        return Image.new('RGB', mol_size, 'white')
    
    # Combine into final reaction diagram
    return _combine_reaction_images(reactant_images, product_images, buyable_indices, target_index, mol_size)

def _process_reactants(rxn, changed_reactants, precursors, buyables, change_color):
    """
    Process reactant molecules: remove atom maps, highlight changes, track buyability.
    
    Returns:
        Tuple of (reactant_data, buyable_indices) where reactant_data is a list of
        (mol, highlight_atoms, atom_colors, original_idx) tuples
    """
    
    reactant_data = []
    original_to_filtered = {}
    
    for i in range(rxn.GetNumReactantTemplates()):
        mol = rxn.GetReactantTemplate(i)
        
        # Convert to clean molecule without atom mapping
        proper_mol, idx_mapping = _remove_atom_mapping(mol)
        
        if proper_mol and not should_exclude_molecule(proper_mol):
            # Highlight atoms that changed
            highlight_atoms, atom_colors = _get_highlights(
                changed_reactants.get(i, []), idx_mapping, proper_mol, change_color
            )
            
            original_to_filtered[i] = len(reactant_data)
            reactant_data.append((proper_mol, highlight_atoms, atom_colors, i))
    
    # Identify which filtered reactants are buyable
    buyable_indices = [
        original_to_filtered[idx] 
        for idx, precursor in enumerate(precursors)
        if precursor in buyables and idx in original_to_filtered
    ]
    
    return reactant_data, buyable_indices


def _process_products(rxn, changed_products, target, change_color):
    """
    Process product molecules: remove atom maps and highlight changes.
    
    Returns:
        Tuple of (product_data, target_index) where:
        - product_data: list of (mol, highlight_atoms, atom_colors) tuples
        - target_index: index of the target product in filtered list, or None
    """

    product_data = []
    original_to_filtered = {}
    target_index = None
    
    for i in range(rxn.GetNumProductTemplates()):
        mol = rxn.GetProductTemplate(i)
        proper_mol, idx_mapping = _remove_atom_mapping(mol)
        
        if proper_mol and not should_exclude_molecule(proper_mol):
            highlight_atoms, atom_colors = _get_highlights(changed_products.get(i, []), idx_mapping, proper_mol, change_color)
            original_to_filtered[i] = len(product_data)
            product_data.append((proper_mol, highlight_atoms, atom_colors))

            if target:
                product_smiles = Chem.MolToSmiles(proper_mol)
                if product_smiles == target:
                    target_index = len(product_data) - 1
    
    return product_data, target_index


def _remove_atom_mapping(mol):
    """
    Convert molecule to clean version without atom mapping numbers.
    
    Returns:
        Tuple of (clean_mol, idx_mapping) where idx_mapping maps old atom indices
        to new atom indices via atom map numbers
    """
    
    # Store mapping: old_idx -> map_num -> new_idx
    idx_to_mapnum = {atom.GetIdx(): atom.GetProp('molAtomMapNumber')
                     for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')}
    
    # Convert through SMILES to get clean molecule
    smiles = Chem.MolToSmiles(mol)
    proper_mol = Chem.MolFromSmiles(smiles)
    
    if not proper_mol:
        return None, {}
    
    # Build reverse mapping: map_num -> new_idx
    mapnum_to_newidx = {atom.GetProp('molAtomMapNumber'): atom.GetIdx() 
                        for atom in proper_mol.GetAtoms() if atom.HasProp('molAtomMapNumber')}
    
    # Remove atom mapping from visualization
    for atom in proper_mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    
    # Combined mapping: old_idx -> map_num -> new_idx
    idx_mapping = {old_idx: mapnum_to_newidx[map_num]
                   for old_idx, map_num in idx_to_mapnum.items()
                   if map_num in mapnum_to_newidx}
    
    return proper_mol, idx_mapping


def _get_highlights(changed_atom_indices, idx_mapping, mol, color):
    """Map changed atom indices to new indices and prepare highlighting."""
    highlight_atoms = []
    atom_colors = {}
    
    for old_idx in changed_atom_indices:
        if old_idx in idx_mapping:
            new_idx = idx_mapping[old_idx]
            if new_idx < mol.GetNumAtoms():
                highlight_atoms.append(new_idx)
                atom_colors[new_idx] = color
    
    return highlight_atoms, atom_colors


def _combine_reaction_images(reactant_images, product_images, buyable_indices, target_index, mol_size):
    """
    Combine individual molecule images into a complete reaction diagram.
    
    Arranges molecules horizontally with:
    - + symbols between reactants/products
    - >> arrow between reactants and products
    - "BUYABLE" labels under buyable reactants
    - "TARGET" label under the target product
    
    Args:
        reactant_images: List of PIL Images for reactants
        product_images: List of PIL Images for products
        buyable_indices: List of indices indicating which reactants are buyable
        target_index: Index of the target product, or None
        mol_size: (width, height) tuple for molecule sizing
    """
    
    label_height = 50
    plus_width = int(mol_size[0] * 0.3)
    arrow_width = int(mol_size[0] * 0.5)
    
    # Calculate total dimensions
    total_width = (
        mol_size[0] * len(reactant_images) + 
        plus_width * (len(reactant_images) - 1) + 
        arrow_width + 
        mol_size[0] * len(product_images) + 
        plus_width * (len(product_images) - 1)
    )
    total_height = mol_size[1] + label_height
    
    # Create canvas
    combined = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(combined)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    x_offset = 0
    
    # Place reactants with + symbols and buyable labels
    for i, img in enumerate(reactant_images):
        combined.paste(img, (x_offset, 0))
        
        # Add "BUYABLE" label
        if i in buyable_indices:
            _draw_centered_text(draw, "BUYABLE", x_offset, mol_size[0], 
                               mol_size[1] + 5, font, 'green')
        
        x_offset += mol_size[0]
        
        if i < len(reactant_images) - 1:
            _draw_plus_sign(draw, x_offset + plus_width // 2, mol_size[1] // 2, 10)
            x_offset += plus_width
    
    # Add >> arrow
    _draw_arrow(draw, x_offset + arrow_width // 2, mol_size[1] // 2, 20)
    x_offset += arrow_width
    
    # Place products with + symbols
    for i, img in enumerate(product_images):
        combined.paste(img, (x_offset, 0))

        # Add "TARGET" label if applicable
        if target_index is not None and i == target_index:
            _draw_centered_text(draw, "TARGET", x_offset, mol_size[0], 
                               mol_size[1] + 5, font, 'red')
            
        x_offset += mol_size[0]
        
        if i < len(product_images) - 1:
            _draw_plus_sign(draw, x_offset + plus_width // 2, mol_size[1] // 2, 10)
            x_offset += plus_width
    
    return combined


def _draw_centered_text(draw, text, x_offset, width, y_pos, font, color):
    """Draw text centered within a given width."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = x_offset + (width - text_width) // 2
    draw.text((text_x, y_pos), text, fill=color, font=font)


def _draw_plus_sign(draw, x_center, y_center, size):
    """Draw a + symbol using two perpendicular rectangles."""
    thickness = size // 5
    # Horizontal bar
    draw.rectangle([x_center - size//2, y_center - thickness//2,
                   x_center + size//2, y_center + thickness//2], fill='black')
    # Vertical bar
    draw.rectangle([x_center - thickness//2, y_center - size//2,
                   x_center + thickness//2, y_center + size//2], fill='black')


def _draw_arrow(draw, x_center, y_center, size):
    """Draw a reaction arrow (→) with a triangular arrowhead."""
    arrow_length = size * 2
    arrow_height = size
    thickness = max(2, size // 10)
    
    # Arrow shaft (horizontal line)
    shaft_start_x = x_center - arrow_length // 2
    shaft_end_x = x_center + arrow_length // 2
    
    draw.line(
        [(shaft_start_x, y_center), (shaft_end_x, y_center)],
        fill='black',
        width=thickness
    )
    
    # Arrowhead (filled triangle)
    head_size = arrow_height // 2
    arrowhead_points = [
        (shaft_end_x, y_center),                           # Tip
        (shaft_end_x - head_size, y_center - head_size//2), # Top
        (shaft_end_x - head_size, y_center + head_size//2)  # Bottom
    ]
    draw.polygon(arrowhead_points, fill='black')

