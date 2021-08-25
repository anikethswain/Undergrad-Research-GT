# Undergrad research @ GT
I worked primarily on 2 different scripts: gp_ICD10_vectors.py and Tree_to_hierarchy.py which automate some preprocessing steps from data in the UK Biobank.

The Tree_to_hierarchy.py script takes a data coding schema used in the UK Biobank and determines all the parent codes given a ICD10 code and creates a file containing this information. This script was based off assumptions for datacoding-19 but is generalizable to other UK Biobank datacoding schema.

The gp_ICD10_vectors.py script went to record level of general practitioner (gp) data and determines the first diagnosis age of diagnosis for patients and then adds this data into a matrix (numpy arrays).
