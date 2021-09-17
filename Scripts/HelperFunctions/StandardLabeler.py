import pandas as pd
import numpy as np

standard_plasmid_class_dict = {"Actinobacteria": 18,
                               "Bacteroidetes": 19,
                               "Chlamydiae": 20,
                               "Cyanobacteria": 21,
                               "Deinococcus-Thermus": 22,
                               "Firmicutes": 23,
                               "Fusobacteria": 24,
                               "other": 25,
                               "Proteobacteria": 26,
                               "Spirochaetes": 27}

standard_chromosome_class_dict = {"Acidobacteria":0,	
                                  "Actinobacteria":1,	
                                  "Bacteroidetes":2,	
                                  "Chlamydiae":3,	
                                  "Chlorobi":4,	
                                  "Chloroflexi":5,	
                                  "Cyanobacteria":6,	
                                  "Deinococcus-Thermus":7,	
                                  "Firmicutes":8,	
                                  "Fusobacteria":9,	
                                  "Nitrospirae":10,	
                                  "other":11,	
                                  "Planctomycetes":12,	
                                  "Proteobacteria":13,	
                                  "Spirochaetes":14,	
                                  "Tenericutes":15,	
                                  "Thermotogae":16,	
                                  "Verrucomicrobia":17}


def standerizeLabels(standard_classes_csv, dataset_path):
    standard_class_df = pd.read_csv(standard_classes_csv, sep='\t')
    plasmid_csv_path = dataset_path + '/final_plasmid_labels.csv'
    chromosome_csv_path = dataset_path + '/final_chromosome_labels.csv'
    plasmid_labels_df = pd.read_csv(plasmid_csv_path)
    chromosome_labels_df = pd.read_csv(chromosome_csv_path)

    plasmid_labels_df["standard_label"] = plasmid_labels_df["Phylum"].apply(
        lambda x: standard_plasmid_class_dict.get(x))
    plasmid_labels_df['standard_label'] = plasmid_labels_df['standard_label'].fillna(
        25.0).astype(int)
    plasmid_labels_df = plasmid_labels_df.merge(
        standard_class_df, on='standard_label', how='inner')

    chromosome_labels_df["standard_label"] = chromosome_labels_df["Phylum"].apply(
        lambda x: standard_chromosome_class_dict.get(x))
    chromosome_labels_df['standard_label'] = chromosome_labels_df['standard_label'].fillna(
        11.0).astype(int)
    chromosome_labels_df = chromosome_labels_df.merge(
        standard_class_df, on='standard_label', how='inner')
    print(plasmid_labels_df)

    print(plasmid_labels_df.Phylum.value_counts())
    print(plasmid_labels_df.standard_class.value_counts())
    print(chromosome_labels_df)

    print(chromosome_labels_df.Phylum.value_counts())
    print(chromosome_labels_df.standard_class.value_counts())


# def second(row):
#     if(row['Phylum'] in standard_plasmid_class_dict.keys()):
#         row['NewLabel'] = standard_plasmid_class_dict[row['Phylum']]
#     else:
#         row['NewLabel'] = standard_plasmid_class_dict['plasmid.other']
