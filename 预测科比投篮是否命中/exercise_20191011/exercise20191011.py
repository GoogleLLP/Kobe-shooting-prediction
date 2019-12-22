# %% 导入包
import pandas as pd
import matplotlib.pyplot as plt


# %% 1.	查看数据集的样本数及特征数，特征的数据类型，以及是否存在缺失值
data = pd.read_csv("../data/kobe_data_20191011.csv")
print(data.shape)
print(data.info())
print(data.isnull().sum())

# %% 2.	统计科比每年的进球数与未进球数，利用条形图可视化，并将结果以文件名“科比1.csv”保存；指出哪年是科比进球最多的年
data["year"] = data["season"].apply(lambda x: int(x[0:4]))
data_new = data[["year", "shot_made_flag"]].copy()
data_new.dropna(inplace=True)
ans = data_new.groupby("year").agg(["sum", "count"])
ans.columns = ans.columns.droplevel()
ans["no"] = ans["count"] - ans["sum"]
ans.drop("count", axis=1, inplace=True)
plt.figure(2)
ans.plot(kind="bar", stacked=True)
plt.savefig("../figures/科比2.png")
plt.show()
ans.to_csv("../data/科比1.csv", index=False)
ans = data["year"].value_counts()
print(ans.idxmax())

# %% 3.	统计1996-2000年、2001-2005年、2006-2010年、2011-2016年所进球数占总进球数的比例，并将结果以文件名“科比2.csv”保存
bins = [1995, 2000, 2005, 2010, 2016]
data["years"] = pd.cut(data["year"], bins, labels=bins[1:])
data_new = data[["shot_made_flag", "years"]].copy()
data_new.dropna(inplace=True)
ans = data_new.groupby("years").sum() / data_new.groupby("years").count()
ans.to_csv("../data/科比2.csv", index=True)


