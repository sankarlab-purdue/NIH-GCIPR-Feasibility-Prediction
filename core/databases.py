import sqlite3
import rdkit.Chem as Chem

# Function to create a new database if needed
def create_database(db_name, table_name, column_definitions):

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        column_str = ", ".join(column_definitions)
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_str});"

        cursor.execute(create_table_query)
        conn.commit()
        print(f"Database '{db_name}' and table '{table_name}' created successfully.")

    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
    finally:
        if conn:
            conn.close()

# Function to canonicalize entries
def canonicalize_smiles(smi:str) -> str:

    try:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except:
        print(f"ERROR: Cannot Canonicalize {smi}")
        canon_smi = smi
    return canon_smi

# Function to add a smile to the database
def add_smile(smiles:str, db:sqlite3.Cursor, table_name:str):
    try:
        canon_smiles = canonicalize_smiles(smiles)
    except:
        canon_smiles = smiles
    db.execute(f"INSERT INTO {table_name} (SMILES) VALUES (?)", (canon_smiles,))
    db.execute("COMMIT")

def add_to_custom(smiles:str, db: sqlite3.Cursor):
    add_smile(smiles, db, 'buyable')

def add_to_excluded(smiles:str, db: sqlite3.Cursor):
    add_smile(smiles, db, 'excluded')

# Function to check if SMILES exists in a database
def smiles_exists(smiles: str, db: sqlite3.Cursor, table_name: str) -> bool:
    canon_smiles = canonicalize_smiles(smiles)
    result = db.execute(f"SELECT 1 FROM {table_name} WHERE SMILES = ? LIMIT 1", (canon_smiles,)).fetchone()
    return result is not None

def is_in_custom(smiles: str, db: sqlite3.Cursor) -> str:
    return f"Molecule {smiles} in Custom: {smiles_exists(smiles, db, 'buyable')}"

def is_in_buyables(smiles: str, db: sqlite3.Cursor) -> str:
    return f"Molecule {smiles} in Buyables: {smiles_exists(smiles, db, 'buyable')}"

def is_in_excluded(smiles: str, db: sqlite3.Cursor) -> str:
    return f"Molecule {smiles} in Excluded: {smiles_exists(smiles, db, 'excluded')}"

# Function to remove SMILES from a database
def remove_smiles(smiles: str, db: sqlite3.Cursor, table_name: str):
    canon_smiles = canonicalize_smiles(smiles)
    db.execute(f"DELETE FROM {table_name} WHERE SMILES = ?", (canon_smiles,))
    db.execute("COMMIT")

def remove_from_custom(smiles: str, db: sqlite3.Cursor):
    remove_smiles(smiles, db, 'buyable')

def remove_from_excluded(smiles: str, db: sqlite3.Cursor):
    remove_smiles(smiles, db, 'excluded')

# Function to sample buyables database
def sample_buyables(sample_size: int, output_file: str=None, buyables_db_path: str='data/split.db', table_name: str='buyable'):
    buyables_conn = sqlite3.connect(buyables_db_path)
    buyables_db = buyables_conn.cursor()

    sampled_smiles = buyables_db.execute(f"SELECT SMILES FROM {table_name} ORDER BY RANDOM() LIMIT ?", (sample_size,)).fetchall()
    smiles_list = [row[0] for row in sampled_smiles]

    if output_file:
        # Create sampled DB
        sample_conn = sqlite3.connect(output_file)
        sample_db = sample_conn.cursor()
        sample_db.execute(f"CREATE TABLE {table_name} (SMILES TEXT)")

        # Insert sampled SMILES
        sample_db.executemany(f"INSERT INTO {table_name} (SMILES) VALUES (?)", [(smiles,) for smiles in smiles_list])
        sample_conn.commit()
        sample_conn.close()
        print(f"Sample saved to {output_file}")

    buyables_conn.close()

def neutralize_smiles(smiles):
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