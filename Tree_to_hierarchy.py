"""This script handles preprocessing coding schema directly from the UK Biobank into a text file that contains each
coding followed by its respective parent codings up till the top most level of the tree

@author Aniketh Swain
@date 6/10/2021
@version 1.0
"""

import pandas as pd
from tqdm import tqdm

def preprocessing_coding_schema(coding_file, end_dir, new_file_name):
    """
    This helper method does the preprocessing of the coding schemas for further usage in data analysis
    List of assumptions (detailed at relevant locations as well):
    # Assumption: No negative codings
    # Assumption: Coding 1 column is presorted in ascending order
    # Assumption: -1 corresponds to the top level tree
    # Assumption: 0 parent_id corresponds to no parent/NULL
    # Assumption: each row is contiguous in the coding schema .tsv file
    """

    coding_df = pd.read_csv(coding_file, sep="\t")
    counter = -1
    # Purpose: Convert repeated -1 codings (if any) into negative distinct codes
    # Assumption: No negative codings
    # Assumption: Coding 1 column is presorted in ascending order
    # Assumption: -1 corresponds to the top level tree
    # Assumption: each row is contiguous in the coding schema .tsv file
    for x in coding_df.index:
        if coding_df.loc[x, "coding"] == -1:
            coding_df.loc[x, "coding"] += counter
            counter += -1
        else:
            break

    # for each code, finding the corresponding parent codes
    # Each row in the dataframe will correspond to each codes upward tree traversal
    # Assumption: 0 parent_id corresponds to no parent/NULL
    # Assumption: All object dtypes are string types

    # Checks for uniqueness of node_ids
    if not coding_df["node_id"].is_unique:
        return -1

    main_row = []
    # iterate through each row
    for x in tqdm(coding_df.index, total=len(coding_df.index)):
        # obtain rows current node_id value and create a temp row list that will be at the end added to the main row
        curr_node = coding_df.at[x, "node_id"]
        row = []
        # curr node becomes zero in the next iteration when the top parent was previously reached
        while curr_node != 0:
            # get the coding value corresponding to the current node and append to current row
            # Assumption: each row is contiguous in the coding schema .tsv file
            temp = coding_df["coding"].where(coding_df["node_id"] == curr_node)
            temp.dropna(inplace=True)
            row.append(temp.iloc[0])

            # update the current node value to its parent node value
            curr_node = coding_df["parent_id"].where(coding_df["node_id"] == curr_node)
            curr_node.dropna(inplace=True)
            curr_node = curr_node.iloc[0]
            # Need to find out how to combine parent_id and coding to be returned in single .where() call if
            # concerned about optimization
        main_row.append(row)

    new_df = pd.DataFrame()
    try:
        new_df = pd.DataFrame(main_row).fillna(0).astype(int)
    except:
        # executes in case of failure to convert values to int, only fills NaN to 0 without any further castings
        new_df = pd.DataFrame(main_row).fillna(0)
    finally:
        # renaming the columns for more user friendly data and finally creating text file
        new_df.rename(columns=lambda x: "parent" + str(x), inplace=True)
        new_df.rename(columns={"parent0": "coding"}, inplace=True)
        new_df.to_csv("{}/{}.txt".format(end_dir, new_file_name), index=False)
