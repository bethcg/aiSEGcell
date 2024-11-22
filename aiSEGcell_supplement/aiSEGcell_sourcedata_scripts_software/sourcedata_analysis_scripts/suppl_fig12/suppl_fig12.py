import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


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

    parser.add_argument('--path_to_csv',
                        type = str,
                        default = './feature_importance_numfeatures_21.csv',
                        help = 'Path to df containtaing feature importances.')
    parser.add_argument('--out_path',
                        type = str,
                        default = '../output/suppl_fig12',
                        help = 'Path to save plot to.')
    
    return parser.parse_args()


def main():
    args = args_parse()
    feature_imp_df = args.path_to_csv
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    df_imp = pd.read_csv(feature_imp_df) 

    # generate color array
    df_imp = df_imp.sort_values(by="importance", ascending=True)
    color = []
    for feature in df_imp["features"].values:
        if ("cell_mask" in feature) or ("focus" in feature) or ("mutlinucleated" in feature):
            color.append("#DED12A")
        else:
            color.append("#F67A7A")

    # generate legend
    custom_lines = [Line2D([0], [0], color='#DED12A', lw=8, label="cell"), Line2D([0], [0], color='#F67A7A', lw=8, label="nucleus")]         

    # feature importance plot    
    plt.rcdefaults()
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=(5.56*cm,8.18*cm))
    plt.rc('font', size=8) #controls default text size

    plt.barh(df_imp.loc[:,'features'], df_imp.loc[:,'importance'], align='center', color = color)
    plt.margins(y=0)
    plt.xlabel('Importance')
    plt.title('Feature importance')
    plt.legend(handles=custom_lines)
    plt.savefig(os.path.join(out_path, f'feature_importance_{len(df_imp)}.pdf'), bbox_inches='tight', dpi=500)
    plt.close()


if __name__ == '__main__':
    main()      
