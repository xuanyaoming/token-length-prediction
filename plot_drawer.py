"""
此文件用于论文作图和信息汇总
"""
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

from matplotlib.pyplot import rcParams

# 设置LaTeX风格的字体
rcParams['font.family'] = 'Times New Roman'
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# 设置图表的尺寸
plt.rcParams["figure.figsize"] = (8, 6)

# 读取数据
# df = pd.read_csv("csvs/IW_4_random_pretrain_dialogs.csv")
# base_dir = "figs/IW_4_trained/random/gripper/"
# df = pd.read_csv("csvs/gripper_random_v2_1.csv")
# base_dir = "figs/gripper_trained/random/IW_1/"
# df = pd.read_csv("csvs/gripper_random_v2_4_train.csv")
# base_dir = "figs/gripper_trained/random/IW_4_train/"
# df = pd.read_csv("csvs/gripper_random_v2_4_test.csv")
# base_dir = "figs/gripper_trained/random/IW_4_test/"
# df = pd.read_csv("gripper_random_v2_1.csv")
# base_dir = "figs/gripper_trained/random/IW_1/"
# df = pd.read_csv("IW_1_fast_v2_1.csv")
# base_dir = "figs/IW_1_trained/fast/IW_1/"
# df = pd.read_csv("IW_1_random_v2_1.csv")
# base_dir = "figs/IW_1_trained/random/IW_1/"
df = pd.read_csv("csvs/IW_4_fast_v2_1.csv")
base_dir = "figs/IW_4_trained/fast/IW_1/"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# 设置主题
sns.set_theme(style="whitegrid")

# 创建并保存误差直方图
ax = sns.histplot(df["diff(pred-real)"], kde=True, color="blue")
ax.set_title(r"The Distribution of Absolute Error")
ax.set_xlabel(r"Absolute Error (predicted value - actual value)")
ax.set_ylabel(r"Count")
plt.savefig(base_dir+"diff.png", bbox_inches='tight')
plt.close()

# 创建并保存误差比例直方图
ax = sns.histplot(df["diff_ratio(diff/real)"], kde=True, color="green")
ax.set_title(r"The Distribution of Relative Error")
ax.set_xlabel(r"Relative Error (absolute error / actual value)")
ax.set_ylabel(r"Count")
plt.savefig(base_dir+"diff_ratio.png", bbox_inches='tight')
plt.close()

# 创建并保存预测EST直方图
ax = sns.histplot(df["est_pred"], kde=True, color="red")
ax.set_title(r"The Distribution of Predicted EST")
ax.set_xlabel(r"Predicted EST")
ax.set_ylabel(r"Count")
plt.savefig(base_dir+"est_pred.png", bbox_inches='tight')
plt.close()

# 创建并保存实际EST直方图
ax = sns.histplot(df["est_real"], kde=True, color="purple")
ax.set_title(r"The Distribution of Actual EST")
ax.set_xlabel(r"Actual EST")
ax.set_ylabel(r"Count")
plt.savefig(base_dir+"est_real.png", bbox_inches='tight')
plt.close()

# 打印误差和误差比例的统计信息
print(df["diff(pred-real)"].std(), df["diff(pred-real)"].mean())
print(df["diff_ratio(diff/real)"].std(), df["diff_ratio(diff/real)"].mean())