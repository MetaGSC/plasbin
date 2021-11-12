import pandas as pd;


def calculateConfutionMatrix(correct_df, incorrect_df):

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

    plasmid_recall = tp_p / (tp_p +tn_p)
    chromosome_recall = tp_c / (tp_c +tn_c)
    plasmid_precision = tp_p / (tp_p +fp_p)
    chromosome_precision = tp_c / (tp_c + fp_c)

    plasmid_f1 = 2*plasmid_recall*plasmid_precision / \
        (plasmid_recall+plasmid_precision)

    chromosome_f1 = 2*chromosome_recall*chromosome_precision / \
        (chromosome_recall+chromosome_precision)

    print('plasmid_recall:-',plasmid_recall)
    print('plasmid_precision:-', plasmid_precision)
    print('plasmid_f1:-', plasmid_f1)
    print('chromosome_recall:-', chromosome_recall)
    print('cromosome_precision:-',chromosome_precision)
    print('chromosome_f1:-', chromosome_f1)


    
