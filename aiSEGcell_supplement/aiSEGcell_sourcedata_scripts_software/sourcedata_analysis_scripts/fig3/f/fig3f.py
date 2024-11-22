import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, wilcoxon


# define output directory
path_out = '../../output/fig3/f'
os.makedirs(path_out, exist_ok=True)

metrics_file = pd.read_csv('./metrics_contain_big_cells.csv')

pretrained = metrics_file['IOU_big_pretrained']
retrained = metrics_file['IOU_big_retrained']

# plt.rcParams["figure.figsize"] = (20,20.44)
plt.rcParams["figure.figsize"] = (20,17.63)
plt.axes().set_yticks([0,1])
plt.ylim(ymin=-0.05, ymax=1.05)
fig = sns.boxplot([pretrained, retrained])
plt.savefig(os.path.join(path_out, 'fig3f.pdf'), dpi=500)

# statistics

# test normal dist with shapiro wilkinson test
pretrained_test = shapiro(pretrained.values)
retrained_test = shapiro(retrained.values)

data = [["pretrained", pretrained_test.statistic, pretrained_test.pvalue, pretrained_test.pvalue>0.05],
        ["retrained", retrained_test.statistic, retrained_test.pvalue, retrained_test.pvalue>0.05]]

df_normality = pd.DataFrame(data, columns=["data", "shapiro_statistics", "shapiro_pvalue", "is_normal_dist"])
df_normality.to_csv(os.path.join(path_out, "normality_test.csv"), index=False)

# perform the Wilcoxon-Signed Rank Test (non-paramteric t-test)
ttest = wilcoxon(pretrained.values, retrained.values)
data_ttest = [["Wilcoxon-Signed Rank Test", ttest.statistic, ttest.pvalue, ttest.pvalue<0.05]]
df_ttest = pd.DataFrame(data_ttest, columns=["test", "W", "pvalue", "significant_difference"])
df_ttest.to_csv(os.path.join(path_out, "wilcoxon.csv"), index=False)
