# %% 导入包
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, chi2, RFE, SelectKBest
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from joblib import load, dump
import tqdm


# %% 投篮区域可视化
def draw_pos(df, column, num=1):
    plt.figure(num)
    plt.title(column)
    area = OrdinalEncoder().fit_transform(df[[column]]).ravel()
    plt.scatter(df["loc_x"], df["loc_y"], alpha=0.05, c=area)
    plt.savefig("../figures/%s投篮区域.jpg" % column)
    plt.show()


# %% 画柱状图对比
def cross_bar(series1, series2):
    cross_tab = pd.crosstab(series2, series1)
    plt.figure()
    cross_tab.plot(kind="bar", stacked=True, legend=len(series1.unique()) < 3, title="count of %s" % series1.name)
    plt.savefig("../figures/%s.jpg" % series1.name)
    plt.show()


# %% 数据清洗
def clean_data(df):
    result = df.drop(["lat", "lon"], axis=1)
    return result


# %% 特征转换
def feature_transform(df):
    result = df.copy()
    result["seconds_remaining"] = df["seconds_remaining"] + 60 * df["minutes_remaining"]
    result["last_5_sec_in_period"] = result["seconds_remaining"].apply(lambda x: 0 if x > 5 else 1)
    result["home_play"] = result["matchup"].apply(lambda x: 1 if x[4] == '@' else 0)
    result["game_date"] = pd.to_datetime(result["game_date"])
    result["game_year"] = result["game_date"].apply(lambda x: x.year)
    result["game_month"] = result["game_date"].apply(lambda x: x.month)
    category_x = 3
    category_y = 3
    result["category_x"] = pd.cut(result["loc_x"], category_x, labels=range(category_x))
    result["category_y"] = pd.cut(result["loc_y"], category_y, labels=range(category_y))
    result.drop(["minutes_remaining", "matchup", "game_date", "loc_x", "loc_y"], axis=1, inplace=True)
    return result


# %% 根据转换后的结果寻找列名
def find_columns(df, x_trans):
    columns = []
    for i in tqdm.tqdm(range(x_trans.shape[1])):
        for j in range(len(df.columns)):
            if (df.iloc[:, j] == x_trans[:, i]).all():
                columns.append(df.iloc[:, j].name)
    return columns


# %% 读取数据
data = pd.read_csv("../data/kobe_data.csv", index_col="shot_id")
print(data.info())

# %% 投篮区域可视化
draw_pos(data, "shot_zone_area", num=1)
draw_pos(data, "shot_zone_basic", num=2)
draw_pos(data, "shot_zone_range", num=3)

# %% 特征game_event_id，game_id，team_name以及team_id与投篮命中预测无关，可以直接舍去
data.drop(["game_event_id", "game_id", "team_name", "team_id"], axis=1, inplace=True)

# %% 划分测试集训练集
data_train = data[data["shot_made_flag"].notnull()]
X_train = data_train.drop("shot_made_flag", axis=1)
y_train = data_train["shot_made_flag"].copy()
data_test = data[data["shot_made_flag"].isnull()]

# %% 查看非数值型变量的分布，正负类样本数据是否平衡，画出正负类样本数据关于各种非数值型变量的统计图
num_column = data_train.select_dtypes(np.number)
str_column = data_train.select_dtypes(np.object)
str_column.apply(cross_bar, args=(y_train, ))

# %% 查看数值型变量的分布，正负类样本数据是否平衡
plt.figure(5)
num_column[num_column["shot_made_flag"] == 0].hist()
plt.savefig("../figures/negative samples num_column.jpg")
plt.show()
plt.figure(6)
num_column[num_column["shot_made_flag"] == 1].hist()
plt.savefig("../figures/positive samples num_column.jpg")
plt.show()

# %% 画出正负类样本数据关于各种数值型变量的箱线图，并分析哪些变量可以有效区分正负类
plt.figure(4)
_, axes = plt.subplots(3, 3, sharey=False)
num_column.boxplot(by="shot_made_flag", ax=axes)
# _, axes = plt.subplots(4, 3, sharey=False)
# axes[3, -1].remove()
# axes[3, -2].remove()
# num_column.boxplot(by="shot_made_flag", ax=axes.ravel()[0:10])
plt.tight_layout()
plt.savefig("../figures/boxplot.jpg")
plt.show()

# %% 基于散点图分析数值型变量两两之间的相关性
plt.figure(7)
scatter_matrix(num_column.drop("shot_made_flag", axis=1))
plt.tight_layout()
plt.savefig("../figures/scatter of num_column.jpg")
plt.show()

# %% 数据清洗
X_train, X_test = clean_data(X_train), clean_data(data_test)

# %% 特征转换
X_train, X_test = feature_transform(X_train), feature_transform(X_test)

# %% 对非数值型变量即名义型特征，进行One-Hot编码
X_train = pd.get_dummies(X_train, columns=[
    "action_type", "combined_shot_type", "season", "shot_type", "shot_zone_area",
    "shot_zone_basic", "shot_zone_range", "opponent"
])
X_test = pd.get_dummies(X_test, columns=[
    "action_type", "combined_shot_type", "season", "shot_type", "shot_zone_area",
    "shot_zone_basic", "shot_zone_range", "opponent"
])
X_test = X_test.reindex(X_train.columns, axis=1)
X_test.fillna(0, inplace=True)

# %% 分别利用方差阈值、随机森林、卡方检验以及RFE方法进行特征选择
thresh = 0.1
trans = VarianceThreshold(threshold=thresh).fit(X_train)
while True:
    X_trans = trans.transform(X_train)
    if X_trans.shape[1] <= 20:
        break
    trans.threshold = trans.threshold + 0.01
X_trans = trans.transform(X_train)
columns_VarianceThreshold = find_columns(X_train, X_trans)

trans = RFE(
    n_features_to_select=20, verbose=1,
    estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=6)
)
X_trans = trans.fit_transform(X_train, y_train)
columns_RandomForest = find_columns(X_train, X_trans)

trans = SelectKBest(chi2, k=20)
X_trans = trans.fit_transform(X_train, y_train)
columns_chi2 = find_columns(X_train, X_trans)

columns = columns_VarianceThreshold + columns_RandomForest + columns_chi2
columns = pd.unique(columns)
X_train = X_train.reindex(columns, axis=1)
X_test = X_test.reindex(columns, axis=1)

# %% 利用PCA来分析前两个主成分
pca = PCA(n_components=2)
X_trans = pca.fit_transform(X_train)
plt.figure(8)
plt.title("PCA")
for label in y_train.unique():
    plt.scatter(X_trans[label == y_train, 0], X_trans[label == y_train, 1], alpha=0.1)
plt.savefig("../figures/PCA.jpg")
plt.show()

# %% 利用AdaBoost，KNN，CART，Naive Bayes，Random forest在训练集上进行3折交叉验证，并基于AUC来比较不同模型的预测效果
models = [KNeighborsClassifier(n_jobs=-1), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_jobs=-1)]
params_KNN = {"n_neighbors": range(3, 10)}
params_DecisionTree = {"max_depth": range(2, 10), "min_samples_leaf": range(5, 100)}
params_NB = {}
params_RandomForest = {"n_estimators": range(2, 70), "max_depth": range(2, 10)}
params = [params_KNN, params_DecisionTree, params_NB, params_RandomForest]
model_name = ["KNN", "Decision Tree", "Naive Bayes", "Random Forest"]
accuracy = []
auc = []
best_params = []
models_best = []
i = 0
for model in tqdm.tqdm(models):
    search = GridSearchCV(
        estimator=model, param_grid=params[i], cv=3, n_jobs=-1, iid=False,
        scoring="accuracy"
    )
    try:
        model = load("../model/%s.pkl" % model_name[i])
    except FileNotFoundError:
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params.append(search.best_params_)
        dump(model, "../model/%s.pkl" % model_name[i])
    models_best.append(model)
    accuracy.append(accuracy_score(y_train, model.predict(X_train)))
    auc.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    i = i + 1

# %% 对测试集进行预测
print(accuracy)
print(auc)
results = np.zeros((X_test.shape[0], 4))
i = 0
for model in models_best:
    results[:, i] = model.predict(X_test)
    i = i + 1
result = results.mean(axis=1) > 0.5
result = result.astype(float)
data_test["shot_made_flag"] = result
data_test.to_csv("../data/predict.csv", columns=["shot_made_flag"])
