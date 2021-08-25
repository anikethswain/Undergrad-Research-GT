"""
Code that determines the first occurrence of a diagnosis for all patients in gp records of UK Biobank

@author Aniketh Swain
@date 8/12/2021
@version 5.0
"""

import pandas as pd
import numpy as np
import h5py
import time
import datetime as dt
# import warnings
#
# from pandas.core.common import SettingWithCopyWarning
#
# warnings.filterwarnings("error", category=SettingWithCopyWarning)

# ------------------------------------------------------
# USER INPUTS

# Global variable file declarations (input locations)
# birthyear_file = "/Volumes/Backup/ukb_phenotypes_mar2021/df_34_birthyear.txt"
# map_path = "/Users/misgut/Repositories/medicalhistory_phenotypes/read_ICD10_map.csv"
# ICD10_hierarchy = "/Volumes/Backup/ukb_phenotypes_mar2021/coding19_hierarchy.txt"
# coding_map = "/Volumes/Backup/ukb_phenotypes_mar2021/coding19.tsv"
# overwrite = False  # TODO update this back to false later after testing

# inputs_filename = "/Volumes/Backup/ukb_phenotypes_mar2021/gp_clinical.txt"
# output_dir = "/Volumes/Backup/ukb_phenotypes_mar2021"
# output_filename = "GP_readtoICD10"  # Without extension
# output_location = "{}/{}.h5".format(output_dir, output_filename)
#
# unavailable_codes_output = "{}/GP_unavailable_keys.txt".format(output_dir)
# available_codes_output = "{}/GP_available_keys.txt".format(output_dir)
# inaccurate_entries_output = "{}/GP_inaccurate_data_entries.txt".format(output_dir)

# contains the birth year for all patients in UKBiobank
birthyear_file = "C:\\GT\\Semester 4.5 (Summer 2021)\\BMED 4699\\Bioinfo research files\\input\\df_34_birthyear.txt"
# mapping file that holds read codes to ICD10 mappings
map_path = "C:\\GT\\Semester 4.5 (Summer 2021)\\BMED 4699\\Bioinfo research files\\maps\\read_ICD10_map.tsv"
# file that holds the ICD10 hierarchy mappings i.e. the parent codes/classifications for each ICD10 code
ICD10_hierarchy = "C:\\GT\\Semester 4.5 (Summer 2021)\\BMED 4699\\Bioinfo research files\\output\\coding19_hierarchy.txt"
# raw ICD10 codes in a tree stucture; the current node and parent node values for an ICD10 code are stored here
coding_map = "C:\\GT\\Semester 4.5 (Summer 2021)\\BMED 4699\\Bioinfo research files\\input\\coding19.tsv"
# overwrite token to prevent/allow overwriting to existing hdf5 files
overwrite = True  # TODO update this back to false later after testing

# gp input file containing all GP patient data
inputs_filename = "C:\\GT\\Semester 4.5 (Summer 2021)\\BMED 4699\\Bioinfo research files\\input\\gp_clinical.txt"
# output directory
output_dir = "C:\\GT\\Semester 4.5 (Summer 2021)\\BMED 4699\\Bioinfo research files\\output"
# output file name for main hdf5 dataset
output_filename = "GP_readtoICD10"  # Without extension
output_location = "{}\\{}.h5".format(output_dir, output_filename)

# meta data that holds information on codes that have and do not have an ICD10 mapping; contains data entries with invalid dates as well
unavailable_codes_output = "{}\\GP_unavailable_keys.txt".format(output_dir)
available_codes_output = "{}\\GP_available_keys.txt".format(output_dir)
inaccurate_entries_output = "{}\\GP_inaccurate_data_entries.txt".format(output_dir)


# -------------------------------------------------------
# FUNCTIONS

def age_code_preprocessing_gp_data(data, mapping_path, patient_id):
    """
    Takes in the raw file containing read codes and returns the patient id with corresponding read-to-ICD10 code

    :param patient_df: raw data file that contains patient_id, date, readV2, readV3, and value 1-3 of a single patient
    :param mapping_path: directory path for the readv2/readv3 to ICD10
    :return: returns final_patient_df and patient_id

    ASSUMPTIONS:
    patient_df has the form as seen in comments below
    dates are stored in %d/%m/%Y format
    codes have the given column names as used
    
    A given record for a patient has either read_v2 or read_v3 but not both because the providers are mutually exclusive and
    provider 3 has read_v3 codes only
    """

    # ASSUMPTION: codes have the given column names as used, # though not a critical assumption as code makes use of indices
    # NOTE there appears to be no actual values for value3 as opening parsed in data into a df has a missing column assumed to be value 3 and hence has been ignored
    # Assumption: patient_df is assumed to have indices as shown below
    # patient_df = patient_df.rename(columns={patient_df.columns[0]: "UKB_ID",
    #                                         patient_df.columns[1]: "data_provider",
    #                                         patient_df.columns[2]: "date",
    #                                         patient_df.columns[3]: "read_2",
    #                                         patient_df.columns[4]: "read_3",
    #                                         patient_df.columns[5]: "value1",
    #                                         patient_df.columns[6]: "value2",
    #                                         patient_df.columns[7]: "value3",
    #                                         })

    # Block here dynamically renames the dataframe with column list as shown below
    columns_list = ["UKB_ID", "data_provider", "date", "read_2", "read_3", "value1", "value2", "value_3"]
    patient_df = pd.DataFrame(data)
    NUMBER_OF_COLUMNS = patient_df.shape[1]
    patient_df.columns = columns_list[:NUMBER_OF_COLUMNS]
    
    patient_df_readv2 = patient_df.loc[:, ("UKB_ID", "date", "read_2")]
    # If there is a read_v3 code as shown by a dataframe dimension greater than or equal to 5, merge the read codes prior to ICD10 mapping
    if patient_df.shape[1] >= 5:
        patient_df_readv3 = patient_df.loc[:, ("UKB_ID", "date", "read_3")]
        
        # Critical Assumption: any single dataprovider uses EITHER read_2 OR read_3. Hence, copying over read_3 into
            # read_2 column does not risk data loss. Known that data provider 3 (Scotland clinics) only uses read v3.
        # Get the read_v3 dataframe non-null values (any row that has a read_v3 code)
        non_null_readv3_df = patient_df_readv3.iloc[np.where(~patient_df_readv3["read_3"].isnull())[0]]["read_3"]
        if len(non_null_readv3_df) != 0:
            # Obtain indices where the read_3 is not null and then take these values and copy into read_2 column
            indices = patient_df_readv2.iloc[np.where(~patient_df_readv3["read_3"].isnull())].index
            patient_df_readv2["read_2"].iloc[indices] = non_null_readv3_df
    
    patient_df = patient_df_readv2.rename(columns={"read_2": "read"})
    map_df = pd.read_csv(mapping_path, sep='\t', header=0, names=["read", "ICD10"])

    # If ICD10 column is NaN, and read is non NaN, that means it is a code that is not in the mapping file
    try:
        # merge ICD 10 codes with read codes (if any) as common keys
        patient_df = patient_df.merge(map_df, on="read", how="left")
        
    except:
        print("Unable to perform a left merge with patient data and read_ICD10 mapping")
        print(patient_df)
        exit(10)

    # Find NaN values in ICD10 corresponding to missing read keys and store in unavailable
    unavailable_keys = patient_df.loc[patient_df["ICD10"].isnull(), "read"].dropna()
    unavailable_keys_list = unavailable_keys.tolist()
    
    # Find mapping values that are available and store in available
    available_keys = patient_df.loc[patient_df["ICD10"].notnull(), "read"].dropna()
    available_keys_list = available_keys.tolist()
    
    # Perform a first occurrence check, convert dates to datetime object to get earliest occurrence
    # ASSUMPTION: dates are stored in %d/%m/%Y format
    # Note: errors="coerece" parameter replaces text with NaT instead of raising an exception, try catch redundant
    patient_df["date"] = pd.to_datetime(patient_df["date"], format='%d/%m/%Y', errors="coerce")
    # removes any NaT (Not a Time) values originating from invalid date-time entries and NaN values in the ICD10 column
    patient_df.dropna(subset=["date", "ICD10"], inplace=True)
    patient_df = patient_df.reset_index(drop=True)

    # drop values where the dates are either 01/01/1901, 02/02/1902, 03/03/1903, or 07/07/2037, check PDF documentation
    unexpected_dates = [dt.datetime(1901, 1, 1), dt.datetime(1902, 2, 2), dt.datetime(1903, 3, 3),
                        dt.datetime(1907, 7, 7)]

    inaccurate_data_entries = patient_df.loc[patient_df["date"].isin(unexpected_dates), "UKB_ID"]
    inaccurate_data_entries = inaccurate_data_entries.to_list()
    
    # drops only the invalid record not the patient
    patient_df = patient_df.drop(patient_df[patient_df["date"].isin(unexpected_dates)].index, axis=0)
    patient_df = patient_df.reset_index(drop=True)

    # Below line to keep only year values
    # Assumption:we are deriving the age based on years itself
    patient_df["date"] = patient_df["date"].dt.year
    # efficiency list is to skip an iteration if the same code is found
    efficiency_list = []
    filtered_patient_df = []
    temp1 = []
    temp2 = []
    for x in patient_df["ICD10"]:
        if x not in efficiency_list:
            temp = patient_df.loc[patient_df["ICD10"] == x, "date"].min()
            filtered_patient_df.append([x, temp])
            # temp1 holds the code
            # temp2 holds the earliest date of occurrence
            temp1.append(x)
            temp2.append(temp)
            efficiency_list.append(x)

    filtered_patient_df = pd.DataFrame({"code": temp1, "date": temp2})
    # Sort all codes into a date ->all codes on that day format
    # for a given year, get all codes and add to that year
    final_patient_df = filtered_patient_df.groupby(["date"])["code"].apply(",".join).reset_index()

    # Converting the found earliest date for each code to age of diagnosis for current patient
    birthyear_df = pd.read_csv(birthyear_file, sep="\t")
    # ASSUMPTION: the column name for the year of birth for a patient is f.34.0.0
    birthyear_df = birthyear_df.rename(columns={"f.34.0.0": "year"})
    
    # drops rows where all elements are missing and convert float types to ints
    # Assumption, data file does not contain any datatype that cannot be safely converted to an int
    birthyear_df = birthyear_df.dropna(how="all")
    birthyear_df = birthyear_df.astype(int)
    birthyear = birthyear_df.loc[birthyear_df["f.eid"] == int(patient_id), "year"].reset_index()
    birthyear = birthyear["year"]

    # apply birth year and subtract from year of diagnosis to get age of diagnosis
    final_patient_df["date"] = final_patient_df["date"].apply(lambda year: int(year) - int(birthyear))
    final_patient_df = final_patient_df.rename(columns={"date": "age"})
    skip = False
    try:
        patient_id = patient_df.iat[0, 0]
    # if the final_patient_df is empty either because there are no ICD10 codes found or if there are no read codes found
    except:
        print("Unable to obtain patient id due no ICD10 codes. Entry may need further inspection")
        skip = True
    finally:
        return skip, final_patient_df, patient_id, unavailable_keys_list, available_keys_list, inaccurate_data_entries


def preprocessing_gp_data(inputs_filename, output_location):
    """
    Description: the main method that starts the preprocessing
    :param inputs_filename: takes in the raw input data
    :param output_location: takes in the location for the processed output data
    :return: null
    """
    # Set up the main dataset in the HDF5 output file
    # start_idx is used to keep track of the line in which the second dataset that keeps track of patient&line in other dataset
    start_idx = 0
    patient_count = 0
    # Open and iterate through the file
    with open(inputs_filename) as file:
        # note 2 readlines called to skip the column name values that are read initially when reading the file
        data_line = file.readline()
        data_line = file.readline()  # first row data
        # Begin iterating through file end terminate once EOF is reached
        while data_line != '':
            data1 = data_line.rstrip().split("\t")  # first row of a patient data
            data1 = [x if x != "" else None for x in data1]
            patient_id = data1[0]
            # get next line data and make sure it is same patient_id then continue this next while loop until it is not
            data_line = file.readline()  # second row read of the patient
            data_line_n = data_line.rstrip().split("\t")  # second row of the patient split
            single_patient_data = [data1]
            # iterate and get all data form one patient and parse into a helper
            while data_line_n[0] == patient_id:
                # data_line_n will be empty only when an empty string i.e. EOF is .split()
                while data_line_n != [] and data_line_n[0] == patient_id:
                    # make any empty strings as None types using list comprehension (more efficient over pure for loop)
                    data_line_n = [x if x != "" else None for x in data_line_n]
                    single_patient_data.append(data_line_n)
                    data_line = file.readline()
                    data_line_n = data_line.rstrip().split("\t")
                    # handle if there are blank lines in between the file due to poorly configured/damaged file
                    # \n is the blank line
                    while data_line == "\n":
                        data_line = file.readline()
                        data_line_n = data_line.rstrip().split("\t")
            
            # call helper function that does further processing, vectorization, and file creation.
            print("Patient count: ", patient_count)
            skip, single_patient_df, ukbid, unavailable_keys_list, available_keys_list, inaccurate_data_entries = age_code_preprocessing_gp_data(
                single_patient_data, map_path, patient_id)
            start_idx, patient_count = vectorization(skip, single_patient_df, ukbid, ICD10_hierarchy, coding_map,
                                                     start_idx,
                                                     output_location, patient_count, unavailable_keys_list,
                                                     available_keys_list, inaccurate_data_entries)
            
def vectorization(skip, age_code_df, UKB_ID, code_hierarchy_map, coding_map, start_idx, output_location, patient_count,
                  unavailable_keys_list, available_keys_list, inaccurate_data_entries):
    """
    Description:
    Total of 5 datasets

    :param start_idx:
    :param output_location:
    :param patient_count:
    :param inaccurate_data_entries:
    :param available_keys_list:
    :param unavailable_keys_list:
    :param age_code_df: the age of diagnosis for a code file
    :param UKB_ID: patient id
    :param code_hierarchy_map: file that holds a code_hierarchy_map hierarchy map
    :param coding_map: data coding19 (ICD10)
    :return: returns start_index and patient_count once a complete append is done

    ASSUMPTIONS:
    All data follows known format of code and age of diagnosis

    """
    
    if skip:
        with open(unavailable_codes_output, "w+") as file:
            for x in unavailable_keys_list:
                file.write("%s\n" % x)

        with open(available_codes_output, "w+") as file:
            for x in available_keys_list:
                file.write("%s\n" % x)

        with open(inaccurate_entries_output, "w+") as file:
            for x in inaccurate_data_entries:
                file.write("%s\n" % x)

        return start_idx, patient_count

    coding_map = pd.read_csv(coding_map, sep="\t")
    code_hierarchy_map = pd.read_csv(code_hierarchy_map)

    coding_map = pd.DataFrame({"idx": coding_map.index,
                               "coding": coding_map["coding"],
                               "meaning": coding_map["meaning"]})

    individuals_idx_df = pd.DataFrame(columns=["UKB ID", "start_idx",
                                               "end_idx"])  # Data frame to keep track of index loc for a ukbid in other dataset
    # create a vector for the patient id
    j = 0
    individual_vectors = []  # A list of the vectors for an individual
    for index, value in age_code_df.iterrows():
        code_list = value[1].rstrip().split(",")  # Split codes into a list by comma
        # code_list = [item.strip("\"") for item in code_list]  # Remove the "" in the string (not always needed here)

        # Add all parent codes to the code_list
        # Assumption: There are no codes labeled "0", since the zeros are place holders. Remove them
        # Combines all parent codes from all ICD-10 code rows, gets rid of duplicates and zeros
        parent_nodes = code_hierarchy_map.iloc[np.where(
            code_hierarchy_map["coding"].isin(code_list))[0]].iloc[:, 1:]
        parent_nodes = np.unique([j for i in parent_nodes.values.tolist() for j in i])
        parent_nodes = np.delete(parent_nodes, np.argwhere(parent_nodes == "0"))
        code_list = code_list + list(parent_nodes)

        # The final codes list should now include all parent codes and child codes, and only unique codes
        # Also, each subsequent date other than the initial one should include all the previous codes
        # NOTE: The parent codes are based on ICD-10 child codes
        code_list = np.unique(np.array(code_list))  # Remove any duplicate codes for the same age

        # Now get the index position for each code in code_list
        indices = []  # Save the indices for each vector in a list
        for current_code in code_list:
            current_index = coding_map.iloc[np.where(coding_map["coding"] == current_code)[0]].index.item()
            indices += [current_index]
        indices = np.array(indices)

        # CHECK
        # Make sure that the length of the indices array is the same as the length of the codes_list
        # If they are not the same with ICD-9 to ICD-10 mapping, it suggests that some ICD-9 codes map to
        # some ICD-10 code or codes that are not in our set of 19k codes.
        if len(code_list) != len(indices):
            print("The length of the codes list is not equal to the length of the indices list. Some ICD-10 codes"
                  "are not being found in the set of 19k ICD-10 codes.")
            exit(1)

        # Get vector using the indices of "1" values
        zeros_array = np.zeros((1, len(code_hierarchy_map)))[0]
        zeros_array[indices] = 1

        # DO NOT make it cumulative. The initial hospital one is cumulative, but for this one
        # we don't need to do that. We can provide only new diagnoses within the year
        # This time, DO NOT add a vector of zeros as the initial value in the list
        individual_vectors += [np.int8(zeros_array)]

        j += 1

        # This is not supposed to be the case for the cancer registry data
    # Being handled in another way

    individual_vectors = np.vstack(individual_vectors).astype(np.int8)
    # Set up the row_index dataset
    # DO NOT add a -1 value to indicate arbitrary zero vector added
    age_list = list(age_code_df["age"])
    age_list = np.array(age_list).reshape(-1, 1)

    # Update the end_idx
    end_idx = start_idx + individual_vectors.shape[0]  # Add the number of rows in the individual_vectors

    # Save the start_idx and end_idx for the individual into a data frame
    individuals_idx_df = individuals_idx_df.append({"UKB ID": np.int64(UKB_ID),
                                                    "start_idx": start_idx,
                                                    "end_idx": end_idx}, ignore_index=True)

    start_idx = end_idx
    with open(unavailable_codes_output, "w+") as file:
        for x in unavailable_keys_list:
            file.write("%s\n" % x)

    with open(available_codes_output, "w+") as file:
        for x in available_keys_list:
            file.write("%s\n" % x)

    with open(inaccurate_entries_output, "w+") as file:
        for x in inaccurate_data_entries:
            file.write("%s\n" % x)
    # Append this to the dataset
    # For the first one, set up the dataset

    if patient_count == 0:
        if overwrite:
            writing = "w"
        else:  # w- creates the file, does not overwrite
            writing = "w-"

        with h5py.File(output_location, writing) as output_file:
            # Set up the main dataset in the HDF5 output file
            # Chunks in the main dataset are 1 x len(col_index_df) and the dataset is resizeable to add rows
            output_file.create_dataset("main_dataset",
                                       data=individual_vectors,
                                       shape=(individual_vectors.shape[0], len(coding_map)),
                                       chunks=(1, len(coding_map)),
                                       maxshape=(None, len(coding_map)),
                                       compression="lzf",
                                       dtype=np.int8)

            # Set up the age dataset in the HDF5 output file
            # Chunks are age_list.shape[0] x 1 (there is one columns, Age) and can add rows
            # Each age corresponds to a row in the main dataset
            output_file.create_dataset("row_index",
                                       data=age_list,
                                       shape=(age_list.shape[0], age_list.shape[1]),
                                       chunks=(1, 1),
                                       maxshape=(None, 1),
                                       compression="lzf",
                                       dtype=np.int8)

            # output_file.create_dataset("unavailable_keys",
            #                            data=unavailable_keys_list,
            #                            shape=(len(unavailable_keys_list), 1),
            #                            chunks=(1, 1),
            #                            maxshape=(None, 1),
            #                            compression="lzf",
            #                            dtype=h5py.string_dtype(encoding='utf-8', length=None),
            #                            )
            # output_file.create_dataset("available_keys",
            #                            data=available_keys_list,
            #                            shape=(len(available_keys_list), 1),
            #                            chunks=(1, 1),
            #                            maxshape=(None, 1),
            #                            compression="lzf",
            #                            dtype=h5py.string_dtype(encoding='utf-8', length=None),
            #                            )
            # output_file.create_dataset("inaccurate_data_entries",
            #                            data=inaccurate_data_entries,
            #                            shape=(len(inaccurate_data_entries), 1),
            #                            chunks=(1, 1),
            #                            maxshape=(None, 1),
            #                            compression="lzf",
            #                            dtype=h5py.string_dtype(encoding='utf-8', length=None),
            #                            )
            # # binary_blob = out.tobytes()
            # For all other individuals, append to the dataset
    else:
        with h5py.File(output_location, "a") as output_file:
            output_file["main_dataset"].resize((
                    output_file["main_dataset"].shape[0] + individual_vectors.shape[0]), axis=0)  # Add rows
            output_file["row_index"].resize((
                    output_file["row_index"].shape[0] + age_list.shape[0]), axis=0)  # Add rows
            # output_file["unavailable_keys"].resize((
            #         output_file["unavailable_keys"].shape[0] + len(unavailable_keys_list)), axis=0)  # Add rows
            # output_file["available_keys"].resize((
            #         output_file["available_keys"].shape[0] + len(available_keys_list)), axis=0)  # Add rows
            # output_file["inaccurate_data_entries"].resize((
            #         output_file["inaccurate_data_entries"].shape[0] + len(inaccurate_data_entries)), axis=0)  # Add rows
            # print("Writing to file...")
            output_file["main_dataset"][-individual_vectors.shape[0]:, :] = individual_vectors
            output_file["row_index"][-age_list.shape[0]:] = age_list
            # output_file["unavailable_keys"][-len(unavailable_keys_list):] = unavailable_keys_list
            # output_file["available_keys"][-len(available_keys_list):] = available_keys_list
            # output_file["inaccurate_data_entries"][-len(inaccurate_data_entries):] = inaccurate_data_entries
            if patient_count % 100 == 0:
                print("i: {}".format(patient_count))
                print("Main Dataset Shape: {}".format(output_file["main_dataset"].shape))
                print("Row Dataset Shape: {}".format(output_file["row_index"].shape))
                print("Elapsed Time: {} m\n".format(round((time.perf_counter() - start_time) / 60)))

    patient_count += 1

    # Save the indexing dataframe to the hdf5 file
    # individuals_idx_df.to_hdf(output_location, key="individual_index_df")

    return start_idx, patient_count

    # output_file = h5py.File("{}/{}.h5".format(output_dir, output_filename),
    #                         "w-")  # w- creates the file, does not overwrite
    # output_location = "{}/{}.h5".format(output_dir, output_filename)


#---------------------- Call function---------------------
start_time = time.perf_counter()
preprocessing_gp_data(inputs_filename, output_location)
end_time = time.perf_counter()
print("total time:", end_time - start_time)
