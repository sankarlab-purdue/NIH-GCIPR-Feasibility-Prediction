# NIH-GCIPR-Feasibility-Prediction

Retrosynthesis planning tools coupled with Feasibility Analyzer, as described in "Enzymatic reaction directional feasibility prediction for computer-aided synthesis planning."

---

## How to Use

1. **Install dependencies**:
   Ensure you have the required Python libraries installed, including `RDKit`, `pandas`, and other dependencies.

   ```bash
   pip install -r requirements.txt

2. **Access the planners**:

    To use the synthesis planners, navigate to one of the following:
    - **Single-step planning**: single_step_planner > UseSingleStepPlanner
    - **Multi-step planning**: multi_step_planner > UseMultiStepPlanner

---

## **Important considerations**:
### Required datasets
Both planners require datasets located in the *data* folder. These datasets are described below. The links to these are already on the notebooks used for the planners, so all you need to do is load the cell before running the analysis. 

- **rdenzyme_db.pkl, retrobiocat_db.pkl:** Templates for RDEnzyme and RetroBioCat reactions
- **buyables.db:** List of commercially available compounds (due to GitHub size limit, please download the database here: [https://purr.purdue.edu/projects/acsfeasiblesynthesis/files](https://purr.purdue.edu/publications/5155/1)) As of now, the repository has a sampled version of this database.
- **custom.db:** User-defined compounds to treat as buyable (editable in data > change_endpoints)
- **excluded.db:** Commercially available compounds that user defined as "Not Buyable" (editable in data > change_endpoints)

### Required input parameters

Provide the following inputs when running either planner:

- **name:** Descriptive name for saving results (multi-step only)
- **target:** SMILES string of the target molecule
- **max_precursors:** Maximum number of most similar reaction to get template and try application with RDEnzyme
- **retrobiocat:** True to include RetroBioCat suggestions, False to exclude them
- **max_depth:** Maximum tree depth to explore (multi-step only)

### Planner outputs

#### Single-step planner
- The retrosynthesis tool generates initial outcomes
- The check_step_feasibility function performs feasibility assessment of each proposed step
- Results are separated into feasible and non-feasible lists and displayed separately
  
#### Multi-step planner
- Results are first displayed in text format
- The parallel_feasibility_pathway_analysis function assesses feasibility of each pathway
- Pathways are classified as feasible or non-feasible and displayed in separate lists
- Results are saved as pickle files in the saved_roots folder

---

## Dataset Builders
A number of datasets had to be built using publicly available data. The builder codes are available on *dataset_builder* folder, and are further described below.

- **build_directionality_dataset**: generates *data\directionality_dataset.pkl*. It is used for feasibility assessment.
- **build_rhea_cosubstrate_dataset**: generates *data\missing_rhea_components.csv*. It is used to generate the full reaction annotation, which is needed for feasibility assessment. 
- **extract_rdenzyme_rhea_templates**: generates *data\rdenzyme_db.pkl*. It contains the pre-extracted templates from RDEnzyme for cache building. 

The folder also contains the code used for leave-one-out cross-validation of the directionality dataset, with the results stored in  the *validation_output* folder. 

If the user wants to only test template extraction and application (*Reaction Chemistry Checker* used in the Feasibility Analyzer), they can refer to *dataset_builder\test_template_application.ipynb*.

## Core files
All of the functions used in the jupyter notebooks are organized in the *core* folder. The core files are further described below. 

- **DirDatasetBuilder**: Has the functions needed to generate the *data\directionality_dataset.pkl* in the *dataset_builder\build_directionality_dataset.ipynb* notebook.
- **FeasibilityAnalyzer**: Has the main functions needed for feasibility assessment of reactions. 
- **ReactChemChecker** and **template_extractor**: has the main functions needed to perform template extraction and application. 
- **BFS**: Has the main functions needed for multi-step synthesis planning, following a breadth-first search. 
- **utils**: Has the retrosynthesis functions (RDEnzyme and RetroBioCat) used in the planners.
- **tree_utils**: Has helper functions for saving, loading and displaying the tree generated during multi-step. 
- **nodes**: Has the description of the Chemical and Reaction Nodes used for tree generation. 



