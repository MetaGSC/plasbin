import pandas as pd;


def calculateConfutionMatrix(correct_df, incorrect_df):

    plasmid_recall = "Undefined"
    plasmid_precision = "Undefined"
    plasmid_f1 = "Undefined"
    chromosome_recall = "Undefined"
    chromosome_precision = "Undefined"
    chromosome_f1 = "Undefined"

    tp_p = len(
        correct_df.loc[(correct_df['label'] ==1 ) & (correct_df['prediction'] ==1)])
    tn_p = len(
        correct_df.loc[(correct_df['label'] ==0 ) & (correct_df['prediction'] ==0)])
    fn_p = len(
        incorrect_df.loc[(incorrect_df['label'] == 1) & (incorrect_df['prediction'] == 0)])
    fp_p = len(
        incorrect_df.loc[(incorrect_df['label'] == 0) & (incorrect_df['prediction'] == 1)])

    tn_c = len(
        correct_df.loc[(correct_df['label'] == 1) & (correct_df['prediction'] == 1)])
    tp_c = len(
        correct_df.loc[(correct_df['label'] == 0) & (correct_df['prediction'] == 0)])
    fp_c = len(
        incorrect_df.loc[(incorrect_df['label'] == 1) & (incorrect_df['prediction'] == 0)])
    fn_c = len(
        incorrect_df.loc[(incorrect_df['label'] == 0) & (incorrect_df['prediction'] == 1)])

    print("tp_p : -", str(tp_p) + "\n", "tn_p : -", str(tn_p) + "\n",
          "fn_p : -", str(fn_p) + "\n", "fp_p : -", str(fp_p) + "\n",
          "tn_c : -", str(tn_c) + "\n", "tp_c : -", str(tp_c) + "\n",
          "fp_c : -", str(fp_c) + "\n", "fn_c : -", str(fn_c) + "\n")

    try:
        plasmid_recall = tp_p / (tp_p +tn_p)
    except:
        print("Zer Division Error Occur for Plasmid Recall")
    try:
        chromosome_recall = tp_c / (tp_c +tn_c)
    except:
        print("Zer Division Error Occur for chromosome_recall")
    try:
        plasmid_precision = tp_p / (tp_p +fp_p)
    except:
        print("Zer Division Error Occur for plasmid_precision")
    try:
        chromosome_precision = tp_c / (tp_c + fp_c)
    except:
        print("Zer Division Error Occur for chromosome_precision")

    try:
        plasmid_f1 = 2*plasmid_recall*plasmid_precision / \
            (plasmid_recall+plasmid_precision)
    except:
        print("Zer Division Error Occur for plasmid_f1")

    try:
        chromosome_f1 = 2*chromosome_recall*chromosome_precision / \
        (chromosome_recall+chromosome_precision)
    except:
        print("Zer Division Error Occur for chromosome_f1")

    print('plasmid_recall:-',plasmid_recall)
    print('plasmid_precision:-', plasmid_precision)
    print('plasmid_f1:-', plasmid_f1)
    print('chromosome_recall:-', chromosome_recall)
    print('cromosome_precision:-',chromosome_precision)
    print('chromosome_f1:-', chromosome_f1)
