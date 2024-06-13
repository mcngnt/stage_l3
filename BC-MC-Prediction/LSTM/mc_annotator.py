import pandas as pd
import os
from tqdm import tqdm

#path = "/Users/abhishekagrawal/corpora/ChiCo/child-caregiver-data/data_cog/Child/AZ/CA-ZN-AN-annotation.csv"

path = "/Users/abhishekagrawal/corpora/ChiCo/child-caregiver-data/data_cog"
filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            # append the file name to the list
            filelist.append(os.path.join(root, file))

child_mc_indicator = {'c-turn': [], 'c-head': ["nodr"], 'c-funch': ["response"], 'speechfunction': ["c-response"]}
parent_mc_indicator = {'p-turn': [], 'p-head': ["nodr"], 'p-funch': ["response"], 'speechfunction': ["p-response"]}
a1_mc_indicator = {'a1-turn': [], 'a1-head': ["nodr"], 'a1-funch': ["response"], 'speechfunction': ["a1-response"]}
a2_mc_indicator = {'a2-turn': [], 'a2-head': ["nodr"], 'a2-funch': ["response"], 'speechfunction': ["a2-response"]}

# with open("./data/extracted_annotations/voice_activity/CA-AN-ZN.f.csv") as fp:
#     parent_annotation = pd.read_csv(fp)
#
# with open("./data/extracted_annotations/voice_activity/CA-AN-ZN.g.csv") as fp:
#     child_annotation = pd.read_csv(fp)
#
# with open(path, "r") as fp:
#     cog_annotation = pd.read_csv(fp, header=None, names=["activity", "participant", "start_time", "start_time_ms",
#                                                          "end_time", "end_time_ms", "time_diff", "time_diff_ms",
#                                                          "annotation",
#                                                          "comment"])


def myround_end(x, base=.05):
    return round(base * round(float(x) / base, 2), 2)


def myround_begin(x, base=.05):
    return round(base * round(float(x) / base), 2)


# for index, row in cog_annotation.iterrows():
#     if row['activity'].strip().lower() in parent_mc_indicator:
#         begin, end = myround_begin(row['start_time_ms']), myround_end(row['end_time_ms'])
#         while begin < end:
#             idx = parent_annotation.index[parent_annotation['frameTimes'] == begin].tolist()[0]
#             if parent_annotation.loc[idx, 'val'] != 1:
#                 parent_annotation.loc[idx, 'val'] = 2
#             begin += 0.05
#             begin = round(begin, 2)
#
#
#     elif row['activity'].strip().lower() in child_mc_indicator:
#         begin, end = myround_begin(row['start_time_ms']), myround_end(row['end_time_ms'])
#         while begin < end:
#             idx = child_annotation.index[child_annotation['frameTimes'] == begin].tolist()[0]
#             if child_annotation.loc[idx, 'val'] != 1:
#                 child_annotation.loc[idx, 'val'] = 2
#             begin += 0.05
#             begin = round(begin, 2)
#
# parent_annotation.to_csv("./data/extracted_annotations/bc_mc_labels/CA-AN-ZN.f_new.csv", index=False)
# child_annotation.to_csv("./data/extracted_annotations/bc_mc_labels/CA-AN-ZN.g_new.csv", index=False)

path = "./data/extracted_annotations/voice_activity/"
for file in tqdm(filelist):
    notation = file.split("/")[-1].split("-")[:-1]
    notation = "-".join(notation)
    with open(file, "r") as fp:
        cog_annotation = pd.read_csv(fp, header=None, names=["activity", "participant", "start_time", "start_time_ms",
                                                             "end_time", "end_time_ms", "time_diff", "time_diff_ms",
                                                             "annotation",
                                                             "comment"])

    with open(path + notation + ".f.csv") as fp:
        parent_annotation = pd.read_csv(fp)

    with open(path + notation + ".g.csv") as fp:
        child_annotation = pd.read_csv(fp)

    for index, row in cog_annotation.iterrows():
        if row['activity'].strip().lower() in parent_mc_indicator:
            begin, end = myround_begin(row['start_time_ms']), myround_end(row['end_time_ms'])
            while begin < end:
                idx = parent_annotation.index[parent_annotation['frameTimes'] == begin].tolist()[0]
                if parent_annotation.loc[idx, 'val'] != 1:
                    parent_annotation.loc[idx, 'val'] = 2
                begin += 0.05
                begin = round(begin, 2)


        elif row['activity'].strip().lower() in child_mc_indicator:
            begin, end = myround_begin(row['start_time_ms']), myround_end(row['end_time_ms'])
            while begin < end:
                idx = child_annotation.index[child_annotation['frameTimes'] == begin].tolist()[0]
                if child_annotation.loc[idx, 'val'] != 1:
                    child_annotation.loc[idx, 'val'] = 2
                begin += 0.05
                begin = round(begin, 2)

        elif row['activity'].strip().lower() in a1_mc_indicator:
            begin, end = myround_begin(row['start_time_ms']), myround_end(row['end_time_ms'])
            while begin < end:
                idx = parent_annotation.index[parent_annotation['frameTimes'] == begin].tolist()[0]
                if parent_annotation.loc[idx, 'val'] != 1:
                    parent_annotation.loc[idx, 'val'] = 2
                begin += 0.05
                begin = round(begin, 2)

        elif row['activity'].strip().lower() in a2_mc_indicator:
            begin, end = myround_begin(row['start_time_ms']), myround_end(row['end_time_ms'])
            while begin < end:
                idx = child_annotation.index[child_annotation['frameTimes'] == begin].tolist()[0]
                if child_annotation.loc[idx, 'val'] != 1:
                    child_annotation.loc[idx, 'val'] = 2
                begin += 0.05
                begin = round(begin, 2)



    parent_annotation.to_csv("./data/extracted_annotations/bc_mc_labels/" + notation + ".f.csv", index=False)
    child_annotation.to_csv("./data/extracted_annotations/bc_mc_labels/" + notation + ".g.csv", index=False)



