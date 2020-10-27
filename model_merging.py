###来自https://zhuanlan.zhihu.com/p/74474886
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import csv
def data_set():
    data_path = r'E:\project\kaggle\data\data.csv'
    data = pd.read_csv(data_path)
    train_label = pd.read_csv(r'E:\project\kaggle\data\label.csv')
    # train_label = data['SalePrice']
    target_variable = np.log(train_label)
    # print(target_variable.shape)
    data = data.drop(['Id'], axis=1)

    # 填充nil
    features_fill_na_none = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
                   'GarageQual','GarageCond','GarageFinish','GarageType',
                   'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
                   'MasVnrType']

    # 填充0
    features_fill_na_0 = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
                          'BsmtFullBath','BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2',
                          'BsmtUnfSF', 'TotalBsmtSF']

    # 填众数
    features_fill_na_mode = ["Functional", "MSZoning", "SaleType", "Electrical",
                             "KitchenQual", "Exterior2nd", "Exterior1st"]

    for feature_none in features_fill_na_none:
        data[feature_none].fillna('None',inplace=True)

    for feature_0 in features_fill_na_0:
        data[feature_0].fillna(0,inplace=True)

    for feature_mode in features_fill_na_mode:
        mode_value = data[feature_mode].value_counts().sort_values(ascending=False).index[0]
        data[features_fill_na_mode] = data[features_fill_na_mode].fillna(mode_value)

    # 用中值代替
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    # 像 Utilities 这种总共才两个值，同时有一个值是作为主要的，这种字段是无意义的，应该删除
    data.drop(['Utilities'], axis=1,inplace=True)
    data_na = (data.isnull().sum() / len(data)) * 100
    data_na.drop(data_na[data_na==0].index,inplace=True)
    data_na = data_na.sort_values(ascending=False)

    data['MSSubClass'] = data['MSSubClass'].apply(str)
    #Changing OverallCond into a categorical variable
    data['OverallCond'] = data['OverallCond'].astype(str)
    #Year and month sold are transformed into categorical features.
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)
    encode_cat_variables = ('Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir',
                            'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence',
                            'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType',
                            'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape',
                            'MSSubClass', 'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood', 'OverallCond', 'PavedDrive',
                            'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'YrSold')

    numerical_features = [col for col in data.columns if col not in encode_cat_variables]
    # for variable in encode_cat_variables:
    #     lbl = LabelEncoder()
    #     lbl.fit(list(data[variable].values))
    #     data[variable] = lbl.transform(list(data[variable].values))

    for variable in data.columns:
        if variable not in encode_cat_variables:
            data[variable] = data[variable].apply(float)
        else:
            data[variable] = data[variable].apply(str)

    data = pd.get_dummies(data)
    # 可以计算一个总面积指标
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    skewed_features = data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_features = skewed_features[abs(skewed_features) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewed_features.shape[0]))

    from scipy.special import boxcox1p
    import csv
    skewed_features_name = skewed_features.index
    lam = 0.15 # 超参数
    for feat in skewed_features_name:
        tranformer_feat = boxcox1p(data[feat], lam)
        data[feat] = tranformer_feat

    data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    data = data.fillna(0)
    with open("test.csv","w",newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(np.array(data))
    return data,target_variable


n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse = np.sqrt(-cross_val_score(model, train, target_variable, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
# score = rmsle_cv(lasso)

KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model) # 复制基准模型，因为这里会有多个模型
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 训练基准模型，基于基准模型训练的结果导出成特征
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y): #分为预测和训练
                # print(train_index,'  ',holdout_index)
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                y_pred = np.array(y_pred)
                y_pred = y_pred.reshape(146)
                print(y_pred.shape)
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 将基准模型预测数据作为特征用来给meta_model训练
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso,model_xgb,model_lgb))

from sklearn.linear_model import LinearRegression
meta_model = LinearRegression()
models = StackingAveragedModels(base_models = (ENet, GBoost, KRR, lasso,model_xgb,model_lgb),
                                                 meta_model = meta_model,
                                                n_folds=10)

# train_train = train[:1000]
# train_label = target_variable[:1000]
# train_test = train[1001:]
# test_label = target_variable[1001:]
# models.fit(train_train,train_label)
# pre = models.predict(train_test)
# averaged_models.fit(train_train,train_label)
# pre2 = averaged_models.predict(train_test)
#
# with open("pre.csv","w",newline = '') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(pre)
#     writer.writerow(pre2)
#     writer.writerow(test_label)
# score1 = rmsle_cv(models)
# score2 = rmsle_cv(averaged_models)
# print('*'*30)
# print(score1)
# print(score2)
if __name__ =='__main__':
    data,target_variable = data_set()
    print(data.shape)
    train = data[:1460]
    test = data[1460:]
    train = np.array(train)
    test = np.array(test)
    target_variable = np.array(target_variable)
    print(train.shape,target_variable.shape)
    averaged_models.fit(train,target_variable)
    pre = averaged_models.predict(test)
    pre = np.exp(pre)
    with open("12.csv","w",newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id','SalePrice'])
        for i in range(1459):
            writer.writerow([i+1461,pre[i]])
    # writer.writerow(train_label)
# print(test)
# score = rmsle_cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
