import pandas as pd


def calculateConfutionMatrix(correct_df, incorrect_df):

    tp_p = len(
        correct_df.loc[(correct_df['label'] == 1) & (correct_df['prediction'] == 1)])
    tn_p = len(
        correct_df.loc[(correct_df['label'] == 0) & (correct_df['prediction'] == 0)])
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

    print('tp_p=', tp_p, 'fp_p=', fp_p, 'fn_p=', fn_p, 'tn_p=', tn_p,
          'tp_c=', tp_c, 'fp_c=', fp_c, 'fn_c=', fn_c, 'tn_c=', tn_c)

    if(tp_p != 0 and tn_p != 0):
        plasmid_recall = tp_p / (tp_p + tn_p)
    else:
        plasmid_recall = 0
        print('plasmid_recall divide by zero')

    if(tp_c != 0 and tn_c != 0):
        chromosome_recall = tp_c / (tp_c + tn_c)
    else:
        chromosome_recall = 0
        print('chromosome_recall divide by zero')

    if(tp_p != 0 and fp_p != 0):
        plasmid_precision = tp_p / (tp_p + fp_p)
    else:
        plasmid_precision = 0
        print('plasmid_precision divide by zero')

    if(tp_c != 0 and fp_c != 0):
        chromosome_precision = tp_c / (tp_c + fp_c)
    else:
        chromosome_precision = 0
        print('chromosome_precision divide by zero')

    if(plasmid_recall != 0 and plasmid_precision != 0):
        plasmid_f1 = 2*plasmid_recall*plasmid_precision / \
            (plasmid_recall+plasmid_precision)
    else:
        plasmid_f1 = 0
        print('plasmid f1 divide by zero')

    if(chromosome_recall != 0 and chromosome_precision != 0):
        chromosome_f1 = 2*chromosome_recall*chromosome_precision / \
            (chromosome_recall+chromosome_precision)
    else:
        chromosome_f1 = 0
        print('chromosome f1 divide by zero')

    print('plasmid_recall:-', plasmid_recall)
    print('plasmid_precision:-', plasmid_precision)
    print('plasmid_f1:-', plasmid_f1)
    print('chromosome_recall:-', chromosome_recall)
    print('cromosome_precision:-', chromosome_precision)
    print('chromosome_f1:-', chromosome_f1)
