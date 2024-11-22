import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import shapiro, ttest_ind, mannwhitneyu


def args_parse():

    '''

        Catches user input.

        Parameters

        ----------
        
        -

        Return

        ------

        Returns a namespace from `argparse.parse_args()`.

    '''

    desc = "Program to generate a bar plot from a feature importance file."

    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--path_to_feature_importance_df',
                        type = str,
                        default = './feature_importance_numfeatures_21.csv',
                        help = 'Path to df containtaing feature importances.')
    
    parser.add_argument('--path_to_df_class',
                        type = str,
                        default = './classification_df_new.csv',
                        help = 'Path to df containtaing data used for classification.')
    
    parser.add_argument('--out_path',
                        type = str,
                        default = '../output/suppl_fig11',
                        help = 'Path to save violin plots.')
    
    return parser.parse_args()

def main():
    args = args_parse()
    feature_imp_df = args.path_to_feature_importance_df
    class_df_path = args.path_to_df_class
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    
    df_class = pd.read_csv(class_df_path)
    
    df_imp = pd.read_csv(feature_imp_df) 
    
    features = df_imp.loc[:,'features']
    
    # violin plot for each feature     
    # delete row where focus feature is nan   
    row_idx = np.where(df_class['focus'].isna())[0]
    df_class.drop(row_idx, inplace = True)

    normality_test_data = []
    ttest_data = []

    # generate violin plot for each feature
    for feat in list(features):
        # split data into class 0 and 1
        dat0 = df_class.loc[df_class['label'] == 0].loc[:,feat]
        dat1 = df_class.loc[df_class['label'] == 1].loc[:,feat]
        
        # generate violin plots
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=(5*3.5*cm,5*1.90*cm))
        violin_parts = plt.violinplot([dat0,dat1], positions=[0,1], showmeans=True, showextrema=True)
        plt.setp(violin_parts['bodies'], facecolor='white', edgecolor='black', alpha=1)
        plt.yticks([min(df_class.loc[:,feat]), max(df_class.loc[:,feat])])
        plt.margins(y=0.05, x=0.1)
        plt.ylabel(feat)
        plt.xlabel('class')

        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = violin_parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        # test normal dist with shapiro wilkinson test
        dat0_test = shapiro(dat0)
        dat1_test = shapiro(dat1)    

        data = [[f"{feat}, class 0", dat0_test.statistic, dat0_test.pvalue, dat0_test.pvalue>0.05],
        [f"{feat}, class 1", dat1_test.statistic, dat1_test.pvalue, dat1_test.pvalue>0.05]]

        normality_test_data = normality_test_data + data

        # t test 
        if dat0_test.pvalue>0.05 and dat1_test>0.05: # if both normal dist perform normal t-test
            ttest = ttest_ind(a=dat0, b=dat1, equal_var=False)
            data_ttest = [[f"{feat}, ttest", ttest.statistic, ttest.pvalue, ttest.pvalue<0.05]]
        else:    
            # perform the Wilcoxon-Mann-Whitney-Test (non-paramteric t-test with unequal n)
            ttest = mannwhitneyu(dat0, dat1)
            data_ttest = [[f"{feat}, Wilcoxon-Mann-Whitney-Test", ttest.statistic, ttest.pvalue, ttest.pvalue<0.05]]

        ttest_data = ttest_data + data_ttest

        textstr = f'class 0 n: {len(dat0)}, class 1 n: {len(dat1)} \nclass 0 normal: {dat0_test.pvalue>0.05} \nclass 1 normal: {dat1_test.pvalue>0.05} \nttest value: {round(ttest.pvalue,5)} \n'    

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        plt.annotate(textstr, xy=(0.6, 0.8), xycoords='axes fraction')
                
        # save in current directory
        plt.savefig(os.path.join(out_path, f'{feat}.pdf'), dpi=300, bbox_inches = "tight")
        plt.close()

    # save noramlity test data
    df_normality = pd.DataFrame(normality_test_data, columns=["data", "shapiro_statistics", "shapiro_pvalue", "is_normal_dist"])
    df_normality.to_csv(os.path.join(out_path, "normality_test.csv"), index=False)    

    # save t test data
    df_ttest = pd.DataFrame(ttest_data, columns=["test", "ttest_statistics", "ttest_pvalue", "significant_difference"])
    df_ttest.to_csv(os.path.join(out_path, "t_test.csv"), index=False)
    
    
if __name__ == '__main__':
    main()           
