import copy
import pickle
import csv
import pandas as pd
import numpy as np
from collections import Counter
import warnings
import ast
import pymysql
from itertools import combinations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wait.parameter import feature_vgap_tn, feature_vgap_n, feature_add, feature_vgap_t, encoder_colums
from Wait.version_txn import get_vgap

warnings.filterwarnings('ignore')


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# 参数1\2\3  新鲜度特征，三个级别，【[db], [tab], [attr], [all]】   _t/_n/_tm 度量方式Time和Num   还有一个扩展的attrextend
# 参数2  额外特征，动态和静态，【[dynamic], [static]】
# 参数3,4  样本数据所在路径、列名
# 暂时没写指定级别和额外特征的构建方式。
# 目前可能组合都构建：
class AccuAwareService:
    def __init__(self, feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data):
        # 仅新鲜度特征，对应需要的的数据列名
        self.db_tn = feature_vgap_tn[0]
        self.tab_tn = feature_vgap_tn[1]
        self.attr_tn = feature_vgap_tn[2]
        self.all_tn = self.db_tn + self.db_tn + self.db_tn
        self.db_t = feature_vgap_t[0]
        self.tab_t = feature_vgap_t[1]
        self.attr_t = feature_vgap_t[2]
        self.all_t = self.db_t + self.db_t + self.db_t
        self.db_n = feature_vgap_n[0]
        self.tab_n = feature_vgap_n[1]
        self.attr_n = feature_vgap_n[2]
        self.all_n = self.db_n + self.db_n + self.db_n
        self.attrextend_tn = ['vtime_park_count',
                              'Q1_1_time', 'Q1_2_time', 'Q1_3_time',
                              'Q2_1_time', 'Q2_2_time', 'Q2_3_time',
                              'Q3_1_time', 'Q3_2_time', 'Q3_3_time',
                              'vnum_park_count',
                              'Q1_1_num', 'Q1_2_num', 'Q1_3_num',
                              'Q2_1_num', 'Q2_2_num', 'Q2_3_num',
                              'Q3_1_num', 'Q3_2_num', 'Q3_3_num']
        self.attrextend_t = ['vtime_park_count',
                            'Q1_1_time', 'Q1_2_time', 'Q1_3_time',
                            'Q2_1_time', 'Q2_2_time', 'Q2_3_time',
                            'Q3_1_time', 'Q3_2_time', 'Q3_3_time']
        self.attrextend_n = ['vnum_park_count',
                            'Q1_1_num', 'Q1_2_num', 'Q1_3_num',
                            'Q2_1_num', 'Q2_2_num', 'Q2_3_num',
                            'Q3_1_num', 'Q3_2_num', 'Q3_3_num']


        self.af = feature_add
        self.mydata = sample_data

        self.eval_result = list()   # modelname/features/RMSE/MAE/R2/function
                                    # 保存评估指标，每种类型的构建完后写入文件，下次使用时清空
                                    # 这个文件在每第一次vgap model时是清空的，后面则是附加


    def get_train_data(self, features):
        x_col = features
        y_col = 'error'

        # 参数array，要分割的数据集，可以是一个或多个数组、矩阵或数据框。通常包括特征矩阵和目标变量。
        # test_size: 测试集的大小，可以是一个浮点数（表示测试集占总数据集的比例）或整数（表示测试集的样本数量）
        # random_size：控制数据分割的随机性。设置相同的种子会产生相同的随机分割结果。
        x_train, x_test, y_train, y_test = \
            train_test_split(self.mydata.loc[:, x_col],
                             self.mydata.loc[:, y_col], test_size=0.2, random_state=42)
        y_test = np.array(y_test).reshape(-1, 1)

        if len(features) == 1:
            x_train = np.array(x_train).reshape(-1, 1)  # 需要是二维
            x_test = np.array(x_test).reshape(-1, 1)

        return x_train, x_test, y_train, y_test

    def evaluation_model(self, y_test, y_pred):
        RMSE = metrics.mean_squared_error(y_test, y_pred) ** 0.5
        MAE = metrics.mean_absolute_error(y_test, y_pred)
        R2 = metrics.r2_score(y_test, y_pred)
        # print("RMSE: %f, MAE: %f, R2: %f" % (RMSE, MAE, R2))
        return [RMSE, MAE, R2]

    def choose_model(self, x_train, x_test, y_train, y_test):
        ### 1.线性回归 ###
        model_linear = LinearRegression()
        # ### 2.岭回归 ###
        # model_ridge = Ridge(alpha=0.8)  # L2正则化，默认01.0
        # # ### 3.Lasso回归 ###
        # model_lasso = Lasso(alpha=0.02, max_iter=100000)  # L1正则化，默认1.0
        # ### 4.Elastic Net回归 ###
        model_elasticnet = ElasticNet()
        # ### 5.决策树回归 ###
        model_decisiontree = DecisionTreeRegressor()
        # ### 6.随机森林回归 ###
        model_randomforest = RandomForestRegressor()  # 数据集相对较小，一般10-100。 n_estimators默认100
        # ### 7.支持向量回归 ###
        model_svr = SVR()
        ### 8.梯度提升回归 ###
        model_gbr = GradientBoostingRegressor()  # 数据集小纬度低，一般50-100  默认100  n_estimators
        ### 9.K最近邻回归 ###
        model_kneighbors = KNeighborsRegressor()  # 默认5  n_neighbors

        model_linear.fit(x_train, y_train)
        # model_ridge.fit(x_train, y_train)
        # model_lasso.fit(x_train, y_train)
        model_elasticnet.fit(x_train, y_train)
        model_decisiontree.fit(x_train, y_train)
        model_randomforest.fit(x_train, y_train)
        model_svr.fit(x_train, y_train)
        model_gbr.fit(x_train, y_train)
        model_kneighbors.fit(x_train, y_train)

        y_pred1 = model_linear.predict(x_test)
        # y_pred2 = model_ridge.predict(x_test)
        # y_pred3 = model_lasso.predict(x_test)
        y_pred4 = model_elasticnet.predict(x_test)
        y_pred5 = model_decisiontree.predict(x_test)
        y_pred6 = model_randomforest.predict(x_test)
        y_pred7 = model_svr.predict(x_test)
        y_pred8 = model_gbr.predict(x_test)
        y_pred9 = model_kneighbors.predict(x_test)

        eval1 = self.evaluation_model(y_test, y_pred1)  # [RMSE, MAE, R2]
        # eval2 = self.evaluation_model(y_test, y_pred2)
        # eval3 = self.evaluation_model(y_test, y_pred3)
        eval4 = self.evaluation_model(y_test, y_pred4)
        eval5 = self.evaluation_model(y_test, y_pred5)
        eval6 = self.evaluation_model(y_test, y_pred6)
        eval7 = self.evaluation_model(y_test, y_pred7)
        eval8 = self.evaluation_model(y_test, y_pred8)
        eval9 = self.evaluation_model(y_test, y_pred9)

        # result_r2 = [eval1[2], eval2[2], eval3[2], eval4[2], eval5[2], eval6[2], eval7[2], eval8[2], eval9[2]]
        # 选择其中前两位RMSE、MAE最小的列表，第三位R2最大的那个
        result_RMSE = [eval1[0], eval4[0], eval5[0], eval6[0], eval7[0], eval8[0], eval9[0]]
        result_MAE = [eval1[1], eval4[1], eval5[1], eval6[1], eval7[1], eval8[1], eval9[1]]
        result_R2 = [eval1[2], eval4[2], eval5[2], eval6[2], eval7[2], eval8[2], eval9[2]]
        index_min_RMSE = result_RMSE.index(min(result_RMSE))
        index_min_MAE = result_MAE.index(min(result_MAE))
        index_max_R2 = result_R2.index(max(result_R2))
        # 找到出现次数最多的项
        result_all = [index_min_RMSE, index_min_MAE, index_max_R2]
        # 使用Counter统计每个元素出现的次数
        result_count = Counter(result_all)
        # 找出次数出现对多的项
        most_common = result_count.most_common(1)
        # 次数最多的项，如果三者一样，以MAE为基准
        if most_common[0][1] == 1:  # 统计次数均1
            index_model = index_min_MAE
        else:
            index_model = most_common[0][0]
        # print(index_model)
        # 选择模型
        if index_model == 0:
            # print("选择了--线性回归")
            return model_linear, [eval1[0], eval1[1], eval1[2], '线性']
        # elif index_max_r2 == 1:
        #     return model_ridge
        # elif index_max_r2 == 2:
        #     return model_lasso
        elif index_model == 1:
            # print("选择了--弹性网络回归")
            return model_elasticnet, [eval4[0], eval4[1], eval4[2], '弹性网络']
        elif index_model == 2:
            # print("选择了--决策树回归")
            return model_decisiontree, [eval5[0], eval5[1], eval5[2], '决策树']
        elif index_model == 3:
            # print("选择了--随机森林回归")
            return model_randomforest, [eval6[0], eval6[1], eval6[2], '随机森林']
        elif index_model == 4:
            # print("选择了--支持向量回归")
            return model_svr, [eval7[0], eval7[1], eval7[2], '支持向量']
        elif index_model == 5:
            # print("选择了--梯度提升回归")
            return model_gbr, [eval8[0], eval8[1], eval8[2], '梯度提升']
        elif index_model == 6:
            # print("选择了--K最近邻回归")
            return model_kneighbors, [eval9[0], eval9[1], eval9[2], '最近邻']


    # attr粒度的特征扩展变换(这个函数暂时没用到）
    # 版本表有三个start、calling、finish
    #            A       B       C
    # gap_table分别记录版本差，比如为：1,5,2
    # 动态特征获取语句有四条  Q1需要start和calling、Q2看start和finish、Q3看calling和finish、Q4看calling
    # 所以对应为           【AB AC BC B】
    # routing_rule四条语句对应0、1路由到备、主，例如 【0,1,0,0】 ，表示Q2去主，其他去备获取数据
    # 用于aware.predict的是转换后的，即【1,5,0,0,5,2,5】
    # 函数：参一：当前版本表记录的结果，参二：当前的路由规则
    def transfer_vgap(self, nowvgap, routing_rule):        # 注意下面这两个参数要跟预测语句和访问属性列的对应关系转换
        cor_num = [2, 2, 2, 1]  # 每个Query中有几个特征
        rule = [0, 1, 0, 2, 1, 2, 1]  # 对应特征的位置。查询语句与动态特征属性值的关联
        full_gap = []
        r = 0
        for i in range(len(routing_rule)):
            for j in range(cor_num[i]):  # 只是用次数，不用这个序号
                if routing_rule[i] == 1:
                    full_gap.append(0)
                else:
                    full_gap.append(nowvgap[rule[r]])
                r += 1

        return full_gap

    # 仅新鲜度特征构造的模型
    # 模型有15个：aware_db、aware_tab、aware_attr、aware_attrextend、aware_all  再附加度量级别_tn/_t/_n
    def build_model_vgap(self):
        self.eval_result = list()  # modelname/features/RMSE/MAE/R2/function

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # db
        x_train, x_test, y_train, y_test = self.get_train_data(self.db_tn)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_db_tn.pkl')
        with open(model_path, 'wb') as file:    # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_db_tn', 'db_tn'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.db_t)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_db_t.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(aware_model, file)
        modelinfo = ['aware_db_t', 'db_t'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.db_n)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_db_n.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(aware_model, file)
        modelinfo = ['aware_db_n', 'db_n'] + evalreault
        self.eval_result.append(modelinfo)
        print("仅新鲜度，db级别，度量timenum、time、num，模型构建完成")

        # tab
        x_train, x_test, y_train, y_test = self.get_train_data(self.tab_tn)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_tab_tn.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(aware_model, file)
        modelinfo = ['aware_tab_tn', 'tab_tn'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.tab_t)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_tab_t.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(aware_model, file)
        modelinfo = ['aware_tab_t', 'tab_t'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.tab_n)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_tab_n.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(aware_model, file)
        modelinfo = ['aware_tab_n', 'tab_n'] + evalreault
        self.eval_result.append(modelinfo)
        print("仅新鲜度，tab级别，度量timenum、time、num，模型构建完成")

        # attr
        x_train, x_test, y_train, y_test = self.get_train_data(self.attr_tn)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_attr_tn.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_attr_tn', 'attr_tn'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.attr_t)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_attr_t.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_attr_t', 'attr_t'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.attr_n)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_attr_n.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_attr_n', 'attr_n'] + evalreault
        self.eval_result.append(modelinfo)
        print("仅新鲜度，attr级别，度量timenum、time、num，模型构建完成")

        # attrextend   这里再函数外的数据预处理就准备好了，为了方便，直接写，不动态转换了
        x_train, x_test, y_train, y_test = self.get_train_data(self.attrextend_tn)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_attrextend_tn.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_attrextend_tn', 'attrextend_tn'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.attrextend_t)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_attrextend_t.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_attrextend_t', 'attrextend_t'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.attrextend_n)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_attrextend_n.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_attrextend_n', 'attrextend_n'] + evalreault
        self.eval_result.append(modelinfo)
        print("仅新鲜度，attrextend级别，度量timenum、time、num，模型构建完成")

        # all level (这时的attr不扩展）
        x_train, x_test, y_train, y_test = self.get_train_data(self.all_tn)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_all_tn.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_all_tn', 'all_tn'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.all_t)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_all_t.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_all_t', 'all_t'] + evalreault
        self.eval_result.append(modelinfo)
        x_train, x_test, y_train, y_test = self.get_train_data(self.all_n)
        aware_model, evalreault = self.choose_model(x_train, x_test, y_train, y_test)
        model_path = os.path.join(script_dir, 'model_pkl', 'aware_all_n.pkl')
        with open(model_path, 'wb') as file:  # 保存模型
            pickle.dump(aware_model, file)
        modelinfo = ['aware_all_n', 'all_n'] + evalreault
        self.eval_result.append(modelinfo)
        print("仅新鲜度，all级别，度量timenum、time、num，模型构建完成")

        # print(self.eval_result)
        # 模型评估结果保存到文件
        with open('/root/dataset_accu/collect/eval_result.csv', 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.eval_result)

    # 有些静态动态特征需要转换，在预处理的时候已经转换了，这里对应换列名
    def replace_cloname(self, base_colname):
        if base_colname == 'now_time':
            return ['now_day', 'now_hour', 'now_minute', 'now_second']
        elif base_colname == 'start_time':
            return ['start_day', 'start_hour', 'start_minute', 'start_second']
        elif base_colname == 'status1_p':
            return ['status1_p_1', 'status1_p_2']
        elif base_colname == 'status1_s':
            return ['status1_s_1', 'status1_s_2']
        elif base_colname == 'status2_p':
            return ['status2_p_1', 'status2_p_2', 'status2_p_3', 'status2_p_4', 'status2_p_5',
                    'status2_p_6', 'status2_p_7', 'status2_p_8', 'status2_p_9', 'status2_p_10']
        elif base_colname == 'status2_s':
            return ['status2_s_1', 'status2_s_2', 'status2_s_3', 'status2_s_4', 'status2_s_5',
                    'status2_s_6', 'status2_s_7', 'status2_s_8', 'status2_s_9', 'status2_s_10']
        elif base_colname == 'warehouse_name':
            return [
                'warehouse_name_1中间板_即热轧板_库', 'warehouse_name_1热轧卷成品库', 'warehouse_name_2中间板_即热轧板_库',
                'warehouse_name_2热轧卷成品库', 'warehouse_name_3中间板_即热轧板_库', 'warehouse_name_3热轧卷成品库',
                'warehouse_name_4中间板_即热轧板_库', 'warehouse_name_4热轧卷成品库', 'warehouse_name_东铁临港库',
                'warehouse_name_中瑞临港库', 'warehouse_name_伟冠临港库', 'warehouse_name_冷轧成品库_10号门',
                'warehouse_name_冷轧成品库_1号门', 'warehouse_name_冷轧成品库_2号门', 'warehouse_name_冷轧成品库_6号门',
                'warehouse_name_冷轧成品库_7号门',  'warehouse_name_冷轧成品库_8号门', 'warehouse_name_冷轧成品库_9号门',
                'warehouse_name_剪切成品库_3号门', 'warehouse_name_剪切成品库_4号门', 'warehouse_name_剪切成品库_5号门',
                'warehouse_name_多头盘螺库', 'warehouse_name_大H型钢成品库', 'warehouse_name_大H型钢成品库_下线库',
                'warehouse_name_大棒中间库', 'warehouse_name_大棒库_一棒', 'warehouse_name_小H型钢成品库',
                'warehouse_name_小H型钢成品库_下线库', 'warehouse_name_小棒库_二棒', 'warehouse_name_小棒库_二棒_下线库',
                'warehouse_name_岚北码头直取库', 'warehouse_name_平整卷成品库_11号门', 'warehouse_name_平整卷成品库_12号门',
                'warehouse_name_平整卷成品库_13号门', 'warehouse_name_开平1_2_成品库', 'warehouse_name_开平3_成品库',
                'warehouse_name_成品中间库', 'warehouse_name_热轧_2150成品_三库', 'warehouse_name_热轧_2150成品_二库',
                'warehouse_name_热轧_2150成品_四库', 'warehouse_name_精整1_成品库', 'warehouse_name_精整2_成品库',
                'warehouse_name_联储卷板', 'warehouse_name_联储线材', 'warehouse_name_运输处临港东库', 'warehouse_name_运输处临港西库',
                'warehouse_name_高线库_一线', 'warehouse_name_高线库_三线', 'warehouse_name_高线库_下线库', 'warehouse_name_高线库_二线']
        else:
            return [base_colname]

    # 在新鲜度特征的基础上增加特
    # 模型有5个：aware_db_addk、aware_tab_addk、aware_attr_addk、aware_attrextend_addk、aware_all_addk
    # 确定添加个数 k，逐一与新鲜度结合，获得度量指标
    # 先静态特征，只要提升的都加入，超过个数就排队，不足用动态补
    # 参数measure，在增加额外特征的时候不需要所有度量级别的基础上进行，可指定 t、n、tn
    ########  注意，保存到文件的数据，新鲜度相关只保留了度量级别和方法，没有具体到版本属性名
    def build_model_vgap_add(self, addtopk, measure):
        # 得到三个基础模型（仅新鲜度）的评估结果
        evaldata = pd.read_csv('/root/dataset_accu/collect/eval_result.csv', header=None,
                               names=['modelname', 'features', 'RMSE', 'MAE', 'R2', 'function'])
        # 度量方式结果所在index
        if measure == 'tn':
            rindex = 0
        elif measure == 't':
            rindex = 1
        else:
            rindex = 2
        # db\tab\attr\attrextend\all  其实也不用所有级别，目前先把所有级别写上，看看效果
        # 这里选着级别暂时不写动态了，手动对这里删减
        base_eval = [[evaldata.loc[rindex+0]['RMSE'], evaldata.loc[rindex+0]['MAE'], evaldata.loc[rindex+0]['R2']],
                     [evaldata.loc[rindex+3]['RMSE'], evaldata.loc[rindex+3]['MAE'], evaldata.loc[rindex+3]['R2']],
                     [evaldata.loc[rindex+6]['RMSE'], evaldata.loc[rindex+6]['MAE'], evaldata.loc[rindex+6]['R2']],
                     [evaldata.loc[rindex+9]['RMSE'], evaldata.loc[rindex+9]['MAE'], evaldata.loc[rindex+9]['R2']],
                     [evaldata.loc[rindex+12]['RMSE'], evaldata.loc[rindex+12]['MAE'], evaldata.loc[rindex+12]['R2']]]
        # print(base_eval)

        # 在 新鲜度级别 的基础上加 额外特征
        if measure == 'tn':
            base_level = [self.db_tn, self.tab_tn, self.attr_tn, self.attrextend_tn, self.all_tn]
        elif measure == 't':
            base_level = [self.db_t, self.tab_t, self.attr_t, self.attrextend_t, self.all_t]
        else:
            base_level = [self.db_n, self.tab_n, self.attr_n, self.attrextend_n, self.all_n]

        for leveli in range(5):
            get_add_feature = list()  # 最后被选择的特征  内容项是： sf/df，name。'RMSE', 'MAE', 'R2', 模型
            need_f = addtopk  # 临时标记，记录每轮可加入的特征数

            # 候选特征的排序，先按个数，再按主备
            oprelable = 'addfeat'
            for feature_i in range(len(self.af)):
                candidate_features = self.af[feature_i]

                # if feature_i == 0:
                #     candidate_features = self.af[0]
                #     oprename = '自带'
                #     oprelable = 'f_user'
                # elif feature_i == 1:
                #     candidate_features = self.af[1]
                #     oprename = '一条到备'
                #     oprelable = 'af_1s'
                # elif feature_i == 2:
                #     candidate_features = self.af[2]
                #     oprename = '一条到主'
                #     oprelable = 'af_1p'
                # elif feature_i == 3:
                #     candidate_features = self.af[3]
                #     oprename = '两条到备'
                #     oprelable = 'af_2s'
                # elif feature_i == 4:
                #     candidate_features = self.af[4]
                #     oprename = '二条到主'
                #     oprelable = 'af_2p'
                # elif feature_i == 5:
                #     candidate_features = self.af[5]
                #     oprename = '十一条到备'
                #     oprelable = 'af_11s'
                # else:
                #     candidate_features = self.af[6]
                #     oprename = '十一条到主'
                #     oprelable = 'af_11p'

                self.eval_result = list()  # 清空，保存运行数据。   moedlname / 加入特征名 / 三个评估结果 / 算法
                temp_inc_eval = list()  # 借用这个变量将所有加一个特征后的结果提升的放入，选择topk。   加入特征名 / 三个评估结果
                for fi in candidate_features:
                    # 特征名虚幻转换和提取
                    replaced_colname = self.replace_cloname(fi)
                    allfeat = base_level[leveli] + replaced_colname
                    x_train, x_test, y_train, y_test = self.get_train_data(allfeat)
                    aware_m, eval_r = self.choose_model(x_train, x_test, y_train, y_test)
                    modelinfo = [oprelable, fi] + eval_r  # 尝试加入一个特征的模型结果记录
                    self.eval_result.append(modelinfo)
                    # 筛选指标提升的---三个指标只要有一个提升的   返回的eval_r时['模型名称计入属性''RMSE', 'MAE', 'R2', '线性']
                    # 注意前两个评估值是越小越好，最后一个越大越好。依次判断，只要有提升就记录这个属性
                    if eval_r[1] < base_eval[leveli][0]:
                        temp_inc_eval.append(modelinfo)
                    elif eval_r[2] < base_eval[leveli][1]:
                        temp_inc_eval.append(modelinfo)
                    elif eval_r[3] > base_eval[leveli][2]:
                        temp_inc_eval.append(modelinfo)
                # 把所有静态尝试加入得到的评估结果记录下来
                with open('/root/dataset_accu/collect/eval_result.csv', 'a+') as file:
                    writer = csv.writer(file)
                    self.eval_result.sort(key=lambda x: x[3])  # 以MAE结果排序，最优在前，默认升序
                    writer.writerows(self.eval_result)
                # print(f"每次加一个 {oprename} 特征的结果")
                print(f"每次加一个特征的结果")
                for i in self.eval_result:
                    print(i)
                print("*****")

                # 加入个数判断
                temp_inc_eval.sort(key=lambda x: x[3])   # 上面排序的是针对有的静态特征，为了记录过程。这只需要对提升指标的进行筛选
                now_flen = len(temp_inc_eval)
                if need_f > now_flen:  # 只要需要加入的 大于 有的，则全部加入
                    get_add_feature.extend(temp_inc_eval)
                    need_f -= now_flen
                else:  # 否则只加入剩余需要的个数
                    get_add_feature.extend(temp_inc_eval[: need_f])
                    need_f = 0
                # 最后判断满足需要加入的特征个数了，跳出循环
                if need_f == 0:
                    break

            # for sf in self.stat:
            #     # 注意如果是now_timne和start_time则需要使用在预处理中变换后的列名
            #     replaced_colname = self.replace_cloname(sf)
            #     allfeat = base_level[leveli] + replaced_colname
            #     x_train, x_test, y_train, y_test = self.get_train_data(allfeat)
            #     aware_m, eval_r = self.choose_model(x_train, x_test, y_train, y_test)
            #     modelinfo = ['sf', sf] + eval_r  # 尝试加入一个特征的模型结果记录
            #     self.eval_result.append(modelinfo)
            #     # 筛选指标提升的---三个指标只要有一个提升的   返回的eval_r时['模型名称计入属性''RMSE', 'MAE', 'R2', '线性']
            #     # 注意前两个评估值是越小越好，最后一个越大越好。依次判断，只要有提升就记录这个属性
            #     if eval_r[1] < base_eval[leveli][0]:
            #         temp_inc_eval.append(modelinfo)
            #     elif eval_r[2] < base_eval[leveli][1]:
            #         temp_inc_eval.append(modelinfo)
            #     elif eval_r[3] > base_eval[leveli][2]:
            #         temp_inc_eval.append(modelinfo)
            # # 把所有静态尝试加入得到的评估结果记录下来
            # with open('/root/dataset_accu/collect/eval_result.csv', 'a+') as file:
            #     writer = csv.writer(file)
            #     self.eval_result.sort(key=lambda x: x[3])  # 以MAE结果排序，最优在前，默认升序
            #     writer.writerows(self.eval_result)
            # print("每次加一个静态特征（备库）的结果")
            # for i in self.eval_result:
            #     print(i)
            # print("*****")
            #
            # # 增加性能的静态特征个数是否够addtopk
            # get_add_feature = list()
            # temp_inc_eval.sort(key=lambda x: x[3])   # 上面排序的是针对有的静态特征，为了记录过程。这只需要对提升指标的进行筛选
            # static_len = len(temp_inc_eval)
            # if static_len >= addtopk:  # 静态满足需要个数
            #     get_add_feature.extend(temp_inc_eval[: addtopk])
            # else:  # 数量不够考虑加入静态主库
            #     get_add_feature.extend(temp_inc_eval)
            #
            #     self.eval_result = list()
            #     temp_inc_eval = list()
            #     for df in self.dyn:
            #         # 注意这里的动态特征有：['status1_p', 'status1_s', 'status2_p', 'status2_s', 'status3_p', 'status3_s']
            #         # 其中一二是list，已经在预处理中拆分开，三是一个值直接使用
            #         replaced_colname = self.replace_cloname(df)
            #         allfeat = base_level[leveli] + replaced_colname
            #         x_train, x_test, y_train, y_test = self.get_train_data(allfeat)
            #         aware_m, eval_r = self.choose_model(x_train, x_test, y_train, y_test)
            #         modelinfo = ['df', df] + eval_r
            #         self.eval_result.append(modelinfo)
            #         # 筛选指标提升的---三个指标只要有一个提升的   返回的eval_r时['RMSE', 'MAE', 'R2', '线性']
            #         if eval_r[1] < base_eval[leveli][0]:
            #             temp_inc_eval.append(modelinfo)
            #         elif eval_r[2] < base_eval[leveli][1]:
            #             temp_inc_eval.append(modelinfo)
            #         elif eval_r[3] > base_eval[leveli][2]:
            #             temp_inc_eval.append(modelinfo)
            #     # 把所有静态尝试加入得到的评估结果记录下来
            #     with open('/root/dataset_accu/collect/eval_result.csv', 'a+') as file:
            #         writer = csv.writer(file)
            #         self.eval_result.sort(key=lambda x: x[3])
            #         writer.writerows(self.eval_result)
            #     print("每次加一个动态特征的结果")
            #     for i in self.eval_result:
            #         print(i)
            #     print("*****")
            #
            #     # 剩余需要加入的特征数量，超过就把能加的加入
            #     need_f = addtopk - static_len
            #     temp_inc_eval.sort(key=lambda x: x[3])
            #     if need_f < len(temp_inc_eval):
            #         get_add_feature.extend(temp_inc_eval[: need_f])
            #     else:
            #         get_add_feature.extend(temp_inc_eval)

            # 现已完成所有特征的选取，开始训练选好的模型了
            colname_all = list()
            for r in get_add_feature:   # get_add_feature的元素项为： sf/df，name。'RMSE', 'MAE', 'R2', 模型
                colname_all.append(r[1])
            # 用这个训练模型，得到 aware_add
            replaced_colname = list()
            # print(colname_all)
            for i in colname_all:
                replace_col = self.replace_cloname(i)
                replaced_colname.extend(replace_col)
            allfeat = base_level[leveli] + replaced_colname
            x_train, x_test, y_train, y_test = self.get_train_data(allfeat)
            aware_m, eval_r = self.choose_model(x_train, x_test, y_train, y_test)

            self.eval_result = list()
            if leveli == 0:
                nowlevel = "db"
            elif leveli == 1:
                nowlevel = "tab"
            elif leveli == 2:
                nowlevel = "attr"
            elif leveli == 3:
                nowlevel = "attrextend"
            else:
                nowlevel = "all"

            print(f"{nowlevel}级别下---------")
            # 记录features时，新鲜度数据只用级别和度量方法记录
            modelname = f"aware_{nowlevel}_{measure}_add{addtopk}"
            vgapname = f"{nowlevel}_{measure}"
            colname_all.insert(0, vgapname)
            modelinfo = [modelname, colname_all] + eval_r
            self.eval_result.append(modelinfo)  # 这是最后选定的模型
            with open('/root/dataset_accu/collect/eval_result.csv', 'a+') as file:
                writer = csv.writer(file)
                writer.writerows(self.eval_result)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            midelpath = f"{modelname}.pkl"
            model_path = os.path.join(script_dir, 'model_pkl', midelpath)
            with open(model_path, 'wb') as file:
                pickle.dump(aware_m, file)

            for i in modelinfo:
                print(i)

    def build_awaremodel(self, addtopk):
        # 仅新鲜度特征
        self.build_model_vgap()
        # 加入额外特征.三个模型。分别在不同级别下的添加
        self.build_model_vgap_add(addtopk, 't')  # 级别写tn、t、n


    # 在使用感知模型时，需要根据模型名确定所需参数，这个函数的作用是将之前训练得到的结果读入，方便查询
    # 数据来自此类中build_awaremodel函数构建模型时存入在/root/dataset_accu/collect/eval_result.csv的数据
    # 结果记在 self.awaremode_And_features： modelname、features、 'RMSE', 'MAE', 'R2', 'function'
    def get_awarefeat_info(self):
        evaldata = pd.read_csv('/root/dataset_accu/collect/eval_result.csv', header=None,
                               names=['modelname', 'features', 'RMSE', 'MAE', 'R2', 'function'])
        # 前12个(不含all的)是仅新鲜度的，直接可用
        # 只要名字里有add的，特征列需要去掉引号，才能是list
        self.awaremode_And_features = evaldata.loc[0:11, 'modelname':'features']
        self.awaremode_And_features['features'] = self.awaremode_And_features['features'].apply(lambda x: [x])
        new_df = pd.DataFrame(columns=['modelname', 'features'])
        for index, row in evaldata.iterrows():
            if '_add' in row['modelname']:
                featurelist = ast.literal_eval(row['features'])
                newrow = {'modelname': row['modelname'], 'features': featurelist}
                new_df = new_df.append(newrow, ignore_index=True)
        self.awaremode_And_features = pd.concat([self.awaremode_And_features, new_df], ignore_index=True)
    # 同上，传入模型名称，只加载会使用到的模型
    def get_awaremodel_info(self, modelnamelist):
        for namei in modelnamelist:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建文件路径
            name = f"{namei}.pkl"
            model_path = os.path.join(script_dir, 'model_pkl', name)
            # modelpath = f"model_pkl/{namei}.pkl"
            with open(model_path, 'rb') as file:
                awaremode = pickle.load(file)
            setattr(self, namei, awaremode)  # 动态创建属性. 名字为self.{namei}，值是导入的aware模型
    # 同上，提前加载归一化和独热编码器pkl
    # 其实可以根据modelnamelist，查询模型信息，只加载被使用的特征，但是有些特征名字是修改后的，后面还会调整
    def get_data_pkl(self, ):
        self.scalers = {}
        self.encoders = {}
        x_all = encoder_colums[:-1]  # 去掉最后一位error名称
        # 只有'warehouse_name'使用的encoder
        x_all.remove('warehouse_name')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for col in x_all:
            name = f"{col}_scaler.pkl"
            model_path = os.path.join(script_dir, 'data_pkl', name)
            with open(model_path, 'rb') as f:
                self.scalers[col] = pickle.load(f)
        model_path = os.path.join(script_dir, 'data_pkl', 'warehouse_name_encoder.pkl')
        with open(model_path, 'rb') as f:
            self.encoders['warehouse_name'] = pickle.load(f)


    # 传入的vgap是筛选度量方式后的。但这里还是要写全可能，因为归一化生成器是固定名称的
    # 得到需要值，根据nowRouting修改版本信息
    # 最后，归一化
    # 这里还是不写动态构建的，因为之前训练aware模型的时候数据预处理有对应类似代码，直接用比较快。有必要后期再改成动态的
    def get_need_vgap(self, awareName, nowVgap, nowRouting):
        # 1. 需要属性扩展的
        needdeat_normal = list()
        origvgap = list()  # attrextend下才会返回，得到的扩展后的原始的vgap，用在finalrouting时使用
        if "attrextend" in awareName:
            # 特征输入顺序
            # ['vtime_park_count', 'Q1_1_time', 'Q1_2_time', 'Q1_3_time', 'Q2_1_time', 'Q2_2_time', 'Q2_3_time', 'Q3_1_time', 'Q3_2_time', 'Q3_3_time',
            #  'vnum_park_count', 'Q1_1_num', 'Q1_2_num', 'Q1_3_num', 'Q2_1_num', 'Q2_2_num', 'Q2_3_num', 'Q3_1_num', 'Q3_2_num', 'Q3_3_num']
            # 对应关系 (可见预处理写的对应关系，得到下面的need_key)
            if '_tn' in awareName or '_t' in awareName:
                needfeat = list()
                need_key = ['vtime_park_count', 'vtime_start_time', 'vtime_entry_time', 'vtime_loading_time',
                            'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_loading_time',
                            'vtime_entry_whouse_time', 'vtime_finish_time', 'vtime_loading_time']
                for key in need_key:
                    needfeat.append(nowVgap[key])
                origvgap.extend(needfeat)
                # 归一化
                needdeat_normal.extend(self.scalers['vtime_park_count'].transform([[needfeat[0]]]))
                needdeat_normal.extend(self.scalers['Q1_1_time'].transform([[needfeat[1]]]))
                needdeat_normal.extend(self.scalers['Q1_2_time'].transform([[needfeat[2]]]))
                needdeat_normal.extend(self.scalers['Q1_3_time'].transform([[needfeat[3]]]))
                needdeat_normal.extend(self.scalers['Q2_1_time'].transform([[needfeat[4]]]))
                needdeat_normal.extend(self.scalers['Q2_2_time'].transform([[needfeat[5]]]))
                needdeat_normal.extend(self.scalers['Q2_3_time'].transform([[needfeat[6]]]))
                needdeat_normal.extend(self.scalers['Q3_1_time'].transform([[needfeat[7]]]))
                needdeat_normal.extend(self.scalers['Q3_2_time'].transform([[needfeat[8]]]))
                needdeat_normal.extend(self.scalers['Q3_3_time'].transform([[needfeat[9]]]))
            if '_tn' in awareName or '_n' in awareName:
                needfeat = list()
                need_key = ['vnum_park_count',
                            'vnum_start_time', 'vnum_entry_time', 'vnum_loading_time',
                            'vnum_entry_time', 'vnum_entry_whouse_time', 'vnum_loading_time',
                            'vnum_entry_whouse_time', 'vnum_finish_time', 'vnum_loading_time']
                for key in need_key:
                    needfeat.append(nowVgap[key])
                origvgap.extend(needfeat)
                # 归一化
                needdeat_normal.extend(self.scalers['vnum_park_count'].transform([[needfeat[0]]]))
                needdeat_normal.extend(self.scalers['Q1_1_num'].transform([[needfeat[1]]]))
                needdeat_normal.extend(self.scalers['Q1_2_num'].transform([[needfeat[2]]]))
                needdeat_normal.extend(self.scalers['Q1_3_num'].transform([[needfeat[3]]]))
                needdeat_normal.extend(self.scalers['Q2_1_num'].transform([[needfeat[4]]]))
                needdeat_normal.extend(self.scalers['Q2_2_num'].transform([[needfeat[5]]]))
                needdeat_normal.extend(self.scalers['Q2_3_num'].transform([[needfeat[6]]]))
                needdeat_normal.extend(self.scalers['Q3_1_num'].transform([[needfeat[7]]]))
                needdeat_normal.extend(self.scalers['Q3_2_num'].transform([[needfeat[8]]]))
                needdeat_normal.extend(self.scalers['Q3_3_num'].transform([[needfeat[9]]]))
            # 根据路由，如果是1，那就是去主获取最新，对应值为 0
            index_set_zero = list()  # 需要被设置为0的索引号
            if len(needdeat_normal) == 10:  # 只有一种度量方法
                if nowRouting[0] == 1:
                    index_set_zero.append(0)
                if nowRouting[1] == 1:
                    index_set_zero.extend([1, 2, 3])
                if nowRouting[2] == 1:
                    index_set_zero.extend([4, 5, 6])
                if nowRouting[3] == 1:
                    index_set_zero.extend([7, 8, 9])
            else:  # 使用了t与n两种度量方式
                if nowRouting[0] == 1:
                    index_set_zero.extend([0, 10])
                if nowRouting[1] == 1:
                    index_set_zero.extend([1, 2, 3, 11, 12, 13])
                if nowRouting[2] == 1:
                    index_set_zero.extend([4, 5, 6, 14, 15, 16])
                if nowRouting[3] == 1:
                    index_set_zero.extend([7, 8, 9, 17, 18, 19])
            needdeat_normal = [0 if i in index_set_zero else value for i, value in enumerate(needdeat_normal)]
        # 2. 不扩展新鲜度值。只会有一种路由可能，全1或全0
        else:
            if all(i == 1 for i in nowRouting):  # 全1，则是确定返回0的个数，也就是不同级别下新鲜度特征个数
                #['vtime_park_count', 'vtime_loading_time', 'vtime_start_time', 'vtime_entry_time',
                # 'vtime_entry_whouse_time', 'vtime_finish_time', 'vtime_queuing_info', 'vtime_cargo_info',
                # 'vtime_warehouse_info', 'vtime_tablea', 'vtime_accuqueuedata']  这个在里面'vtime_tablea'，但是不属于特征
                level = ['all_tn', 'all_t', 'all_n', 'db_tn', 'db_t', 'db_n',
                         'tab_tn', 'tab_t', 'tab_n', 'attr_tn', 'attr_t', 'attr_n']
                level_num = {'all_tn': 20, 'all_t': 10, 'all_n' : 10, 'db_tn': 2, 'db_t': 1, 'db_n': 1,
                             'tab_tn': 6, 'tab_t': 3, 'tab_n': 3, 'attr_tn':12, 'attr_t': 6, 'attr_n': 6}
                nowlevel = ""
                for k in level:
                    if k in awareName:
                        nowlevel = k
                        break
                needdeat_normal = [0] * level_num[nowlevel]
            else:  # 全0，返回需要的版本差
                # vgap_value = list(nowVgap.values())  # 得到原始值，所有级别的都在
                # 根据级别和度量筛选特征并归一化。 （注意这个顺序是训练模型时的顺序）
                if "db" in awareName or "all" in awareName:
                    needfeat = list()
                    need_key = list()
                    key_t = ['vtime_accuqueuedata']
                    key_n = ['vnum_accuqueuedata']
                    if "_t" in awareName or "_tn" in awareName:
                        need_key.extend(key_t)
                    if "_n" in awareName or "_tn" in awareName:
                        need_key.extend(key_n)
                    # 获取版本差值
                    for key in need_key:
                        needfeat.append(nowVgap[key])
                    # 归一化
                    if "_t" in awareName or "_tn" in awareName:
                        needdeat_normal.extend(self.scalers['vtime_accuqueuedata'].transform([[needfeat[0]]]))
                    if "_n" in awareName or "_tn" in awareName:
                        if "_n" in awareName:
                            data_i = 0
                        else:
                            data_i = 1
                        needdeat_normal.extend(self.scalers['vnum_accuqueuedata'].transform([[needfeat[data_i]]]))
                if "tab" in awareName or "all" in awareName:
                    needfeat = list()
                    need_key = list()
                    key_t = ['vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info']
                    key_n = ['vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info']
                    if "_t" in awareName or "_tn" in awareName:
                        need_key.extend(key_t)
                    if "_n" in awareName or "_tn" in awareName:
                        need_key.extend(key_n)
                    # 获取版本差值
                    for key in need_key:
                        needfeat.append(nowVgap[key])
                    # 归一化
                    if "_t" in awareName or "_tn" in awareName:
                        needdeat_normal.extend(self.scalers['vtime_queuing_info'].transform([[needfeat[0]]]))
                        needdeat_normal.extend(self.scalers['vtime_cargo_info'].transform([[needfeat[1]]]))
                        needdeat_normal.extend(self.scalers['vtime_warehouse_info'].transform([[needfeat[2]]]))
                    if "_n" in awareName or "_tn" in awareName:
                        if "_n" in awareName:
                            data_i = [0, 1, 2]
                        else:
                            data_i = [3, 4, 5]
                        needdeat_normal.extend(self.scalers['vtime_queuing_info'].transform([[needfeat[data_i[0]]]]))
                        needdeat_normal.extend(self.scalers['vtime_cargo_info'].transform([[needfeat[data_i[1]]]]))
                        needdeat_normal.extend(self.scalers['vtime_warehouse_info'].transform([[needfeat[data_i[2]]]]))
                if "_attr_" in awareName:
                    needfeat = list()
                    need_key = list()
                    key_t = ['vtime_park_count', 'vtime_loading_time', 'vtime_start_time', 'vtime_entry_time',
                                    'vtime_entry_whouse_time', 'vtime_finish_time']
                    key_n = ['vnum_park_count', 'vnum_loading_time', 'vnum_start_time', 'vnum_entry_time',
                                    'vnum_entry_whouse_time', 'vnum_finish_time']
                    if "_t" in awareName or "_tn" in awareName:
                        need_key.extend(key_t)
                    if "_n" in awareName or "_tn" in awareName:
                        need_key.extend(key_n)
                    # 获取版本差值
                    for key in need_key:
                        needfeat.append(nowVgap[key])
                    # 归一化
                    if "_t" in awareName or "_tn" in awareName:
                        needdeat_normal.extend(self.scalers['vtime_park_count'].transform([[needfeat[0]]]))
                        needdeat_normal.extend(self.scalers['vtime_loading_time'].transform([[needfeat[1]]]))
                        needdeat_normal.extend(self.scalers['vtime_start_time'].transform([[needfeat[2]]]))
                        needdeat_normal.extend(self.scalers['vtime_entry_time'].transform([[needfeat[3]]]))
                        needdeat_normal.extend(self.scalers['vtime_entry_whouse_time'].transform([[needfeat[4]]]))
                        needdeat_normal.extend(self.scalers['vtime_finish_time'].transform([[needfeat[5]]]))
                    if "_n" in awareName or "_tn" in awareName:
                        if "_n" in awareName:
                            data_i = [0, 1, 2, 3, 4, 5]
                        else:
                            data_i = [6, 7, 8, 9, 10, 11]
                        needdeat_normal.extend(self.scalers['vnum_park_count'].transform([[needfeat[data_i[0]]]]))
                        needdeat_normal.extend(self.scalers['vnum_loading_time'].transform([[needfeat[data_i[1]]]]))
                        needdeat_normal.extend(self.scalers['vnum_start_time'].transform([[needfeat[data_i[2]]]]))
                        needdeat_normal.extend(self.scalers['vnume_entry_time'].transform([[needfeat[data_i[3]]]]))
                        needdeat_normal.extend(self.scalers['vnum_entry_whouse_time'].transform([[needfeat[data_i[4]]]]))
                        needdeat_normal.extend(self.scalers['vnum_finish_time'].transform([[needfeat[data_i[5]]]]))

        return needdeat_normal, origvgap   # 第二个返回值是原始的扩展后的值，如果为空，则不是attrextend类型

    # 在预测的时候需要将时间信息进行提取
    def get_timeinfo(self, timevalue):
        timeinfo = list()
        timeinfo.append(timevalue.day)
        timeinfo.append(timevalue.hour)
        timeinfo.append(timevalue.minute)
        timeinfo.append(timevalue.second)
        return timeinfo

    # routing阶段调用的，对路由长度eouting_len，也就是请求中查询个数，根据需求的被指定到主的数量，的可能数量
    ####  注意注意。这里调整一下，不能按顺序生成，要按对应位置的查询数据，查询数少的优先
    def possible_routing(self, routing_len, num_to_p):
        # 1. 先得到可能得路由顺序
        def routing_candidate():
            base_list = [0] * routing_len
            for indices in combinations(range(routing_len), num_to_p):
                new_list = base_list[:]
                for index in indices:
                    new_list[index] = 1
                yield new_list
        r_candidate = list(routing_candidate())
        # 2. 算总的查询数据
        rule_querynum = [1, 17, 7, 1.5]
        # 计算各个可能路由对应的值
        values = [sum([rule_querynum[i] for i in range(4) if r_candidate[j][i] == 1]) for j in range(len(r_candidate))]
        # 对possible_r内部的列表按值排序
        sorted_possible_r = [r for _, r in sorted(zip(values, r_candidate))]
        return sorted_possible_r



    # routing路由阶段，对最后路由到的结果，即使到主也可以再判断原本vgap为0的，使其路由到备库
    # 想使用这个功能，必须有attr级别的度量结果，因为用了这个才可以判断原本vgap组合起来是否为0，可以到主库
    def get_final_routing(self, origvgap, newrouting):
        # 有四个查询，分别对应的个数据为1,3,3,3
        # ['vtime_park_count',
        #  'vtime_start_time', 'vtime_entry_time', 'vtime_loading_time',
        #  'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_loading_time',
        #  'vtime_entry_whouse_time', 'vtime_finish_time', 'vtime_loading_time']

        finalrouting = copy.copy(newrouting)
        # 如果为10长度，表示只有一种度量方式
        if len(origvgap) == 10:
            if newrouting[0] == 1:  # 只有有“1”才判断是否可能是没有版本差的，转到备库
                if origvgap[0] == 0.0:
                    finalrouting[0] = 0
            if newrouting[1] == 1:
                if origvgap[1] == 0.0 and origvgap[2] == 0.0 and origvgap[3] == 0.0:
                    finalrouting[1] = 0
            if newrouting[2] == 1:
                if origvgap[4] == 0.0 and origvgap[5] == 0.0 and origvgap[6] == 0.0:
                    finalrouting[2] = 0
            if newrouting[3] == 1:
                if origvgap[7] == 0.0 and origvgap[8] == 0.0 and origvgap[9] == 0.0:
                    finalrouting[3] = 0
        # else:
            # 目前不会使用两种度量方式在对比环节

        return finalrouting

    # 传入需要的模型特征名，得到值。并归一化及编码
    def get_code_data(self, original_addfeat,  now_time, start_time, warehouse_name):
        need_addfeat = list()
        pdb = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                              database='accuqueuedata', charset='utf8')
        sdb = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                              database='accuqueuedata', charset='utf8')
        cursor_p = pdb.cursor()
        cursor_s = sdb.cursor()
        # 对每个额外变量，需要进行处理，然后再归一化
        for addi in original_addfeat:
            if addi == "now_time":
                now_time_part = self.get_timeinfo(now_time)  # 返回list，天、小时、分钟、秒
                need_addfeat.extend(self.scalers['now_day'].transform([[now_time_part[0]]]))
                need_addfeat.extend(self.scalers['now_hour'].transform([[now_time_part[1]]]))
                need_addfeat.extend(self.scalers['now_minute'].transform([[now_time_part[2]]]))
                need_addfeat.extend(self.scalers['now_second'].transform([[now_time_part[3]]]))
            elif addi == "start_time":
                start_time_part = self.get_timeinfo(start_time)  # 返回list，天、小时、分钟、秒
                need_addfeat.extend(self.scalers['start_day'].transform([[start_time_part[0]]]))
                need_addfeat.extend(self.scalers['start_hour'].transform([[start_time_part[1]]]))
                need_addfeat.extend(self.scalers['start_minute'].transform([[start_time_part[2]]]))
                need_addfeat.extend(self.scalers['start_second'].transform([[start_time_part[3]]]))
            elif addi == "warehouse_name":
                encode_data = self.encoders['warehouse_name'].transform([[warehouse_name]])
                need_addfeat.extend(encode_data.toarray().flatten())  # 将稀疏矩阵转换为一维数组
            elif addi == "park_count_p" or addi == "park_count_s":
                sql = "SELECT park_count FROM warehouse_info WHERE warehouse_name = %s;"
                if addi == "park_count_p":
                    cursor_p.execute(sql, (warehouse_name,))
                    r = cursor_p.fetchone()[0]
                    need_addfeat.append(self.scalers['park_count_p'].transform([[r]]))
                else:
                    cursor_s.execute(sql, (warehouse_name,))
                    r = cursor_s.fetchone()[0]
                    need_addfeat.append(self.scalers['park_count_s'].transform([[r]]))
            elif addi == "status1_p" or addi == "status1_s" or addi == "status2_p" or addi == "status2_s":
                # 区分状态1和状态2的sql语句
                if addi == "status1_p" or addi == "status1_s":
                    sql = "SELECT cargo_name, recordid FROM queuing_info WHERE warehouse_name = %s " \
                          "AND start_time < %s AND " \
                          "(entry_time IS NULL OR entry_time > %s);"  # 加一个recordid后期没用，只是一开始测试使用
                else:  # status2_p  status2_s
                    sql = "SELECT cargo_name, recordid FROM queuing_info WHERE warehouse_name = %s " \
                          "AND start_time < %s AND entry_time IS NOT NULL AND entry_time <= %s AND" \
                          "(entry_whouse_time IS NULL OR entry_whouse_time > %s);"
                # 区分属性是主还是备库
                if addi == "status1_p" or addi == "status2_p":
                    cursor_p.execute(sql, (warehouse_name, start_time, now_time))
                    r = cursor_p.fetchall()
                else:
                    cursor_s.execute(sql, (warehouse_name, start_time, now_time))
                    r = cursor_s.fetchall()
                # 得到结果，获取车辆的货物加载时间
                loadingtime_list1 = list()  # 他们的loading时间，后续算等待时间会用到
                if r:  # 如果没有结果，前面没车。否则计算前面车数及他们所需工作时间
                    for t in r:
                        sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
                        if addi == "status1_p" or addi == "status2_p":
                            cursor_p.execute(sql, (t[0],))
                            r = cursor_p.fetchall()
                        else:
                            cursor_s.execute(sql, (t[0],))
                            r = cursor_s.fetchall()
                        loadingtime_list1.append(r)
                # 结果需要补足十位数
                if len(loadingtime_list1) < 10:
                    processed_list = loadingtime_list1[:10] + [0] * (10 - len(loadingtime_list1))
                else:
                    processed_list = loadingtime_list1[:10]
                # 归一化
                data_1 = f"{addi}_1"
                data_2 = f"{addi}_2"
                data_3 = f"{addi}_3"
                data_4 = f"{addi}_4"
                data_5 = f"{addi}_5"
                data_6 = f"{addi}_6"
                data_7 = f"{addi}_7"
                data_8 = f"{addi}_8"
                data_9 = f"{addi}_9"
                data_10 = f"{addi}_10"
                need_addfeat.append(self.scalers[data_1].transform([[processed_list[0]]]))
                need_addfeat.append(self.scalers[data_2].transform([[processed_list[1]]]))
                need_addfeat.append(self.scalers[data_3].transform([[processed_list[2]]]))
                need_addfeat.append(self.scalers[data_4].transform([[processed_list[3]]]))
                need_addfeat.append(self.scalers[data_5].transform([[processed_list[4]]]))
                need_addfeat.append(self.scalers[data_6].transform([[processed_list[5]]]))
                need_addfeat.append(self.scalers[data_7].transform([[processed_list[6]]]))
                need_addfeat.append(self.scalers[data_8].transform([[processed_list[7]]]))
                need_addfeat.append(self.scalers[data_9].transform([[processed_list[8]]]))
                need_addfeat.append(self.scalers[data_10].transform([[processed_list[9]]]))

            elif addi == "status3_p" or addi == "status3_s":
                sql = "SELECT cargo_name, entry_whouse_time, recordid FROM queuing_info WHERE warehouse_name = %s " \
                      "AND start_time < %s AND entry_whouse_time IS NOT NULL AND entry_whouse_time <= %s AND" \
                      "(finish_time IS NULL OR finish_time > %s);"
                if addi == "status3_p":
                    cursor_p.execute(sql, (warehouse_name, start_time, now_time, now_time))
                    r = cursor_p.fetchone()
                else:
                    cursor_s.execute(sql, (warehouse_name, start_time, now_time, now_time))
                    r = cursor_s.fetchone()
                remain_loadingtime = 0
                if r:
                    sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
                    if addi == "status3_p":
                        cursor_p.execute(sql, (r[0],))
                        need_loadtime = cursor_p.fetchone()[0]
                    else:
                        cursor_s.execute(sql, (r[0],))
                        need_loadtime = cursor_s.fetchone()[0]
                    worked_time = (now_time - r[1]).total_seconds() / 60  # 计算已经工作时间
                    remain_loadingtime = need_loadtime - worked_time
                    if remain_loadingtime < 0:
                        remain_loadingtime = 0
                if addi == "status3_p":
                    need_addfeat.append(self.scalers['status3_p'].transform([[remain_loadingtime]]))
                else:
                    need_addfeat.append(self.scalers['status3_s'].transform([[remain_loadingtime]]))

        cursor_p.close()
        cursor_s.close()
        pdb.close()
        sdb.close()

        return need_addfeat


    # 根据指定模型名，和路由方向，获取特征及处理转换，得到可能损失
    def predict(self, awareName, needfeat_all, nowRouting, nowVgap_dict, now_time, start_time, warehouse_name):
        # 第一个是新鲜度级别名，如果没有第二个值，那么这个模型就是只用新鲜度构建的模型
        # 1. 新鲜度特征   （需要扩展，或者直接将字典转为list使用）
        need_vgap, origvgap = self.get_need_vgap(awareName, nowVgap_dict, nowRouting)

        # 2. 额外特征
        original_addfeat = needfeat_all[1:]  # 第0位写的是级别
        need_addfeat = self.get_code_data(original_addfeat, now_time, start_time, warehouse_name)

        # 3. 根据模型名获取模型得到预测结果
        if hasattr(self, awareName):
            # 将稀疏矩阵转换为密集矩阵，并且使用 flatten() 方法将其展平为一维数组，然后再拼接到归一化后的特征中
            need_all_features = need_vgap + need_addfeat
            aware_model = getattr(self, awareName)
            input_data = np.array(need_all_features).reshape(1, -1)
            loss = aware_model.predict(input_data)
            return loss[0], origvgap
        else:
            print("感知模型中没有这个模型")
            return 0, []


    # 路由策略
    # 感知模型通过名字指定即可，已经提前导入pkl动态生成存放对应模型的变量了
    # 传入的当前版本信息是根据名字中带有的度量方法筛选后的结果。是个字典
    # 顺序是time为例 ['vtime_park_count', 'vtime_loading_time',
    #                'vtime_start_time', 'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_finish_time',
    #                'vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info', 'vtime_tablea',
    #                'vtime_accuqueuedata']
    def routing(self, tolerable_loss, awareName, nowVgap_dict, now_time, start_time, warehouse_name):
        # 此指定的感知模型awareName，所需要的特征项名
        filtered_df = self.awaremode_And_features[self.awaremode_And_features['modelname'] == awareName]
        needfeat_all = filtered_df.iloc[0]['features']

        # 这里对应用户模型需要的，停车数、状态1、2、3的路由方向。分别包含的sql语句数是1、3、3、3
        nowRouting = [0, 0, 0, 0]
        # 全部路由到备库时的可能损失.
        nowLoss, origvgap = self.predict(awareName, needfeat_all,
                                         nowRouting, nowVgap_dict, now_time, start_time, warehouse_name)

        # 路由决策
        if nowLoss < 1.0:
            nowLoss = 0.0    # 预测出来的可容忍损失loss是小于1的（也就是60以下），那就当做是没有损失
        beginLoss = nowLoss  # 创建这个变量是为了在uni_diff实验中，返回最开始的可能loss来判断uni方法。不然的话这里最后返回的newLoss是diff方法路径变化后的
        if nowLoss <= tolerable_loss:  # 满足需求则全部直接路由到备份
            newLoss = nowLoss
            newRouting = nowRouting
            finalRouting = nowRouting
            return beginLoss, newLoss, newRouting, finalRouting

        else:  # 容忍度不满足需求
            if 'attrextend' in needfeat_all[0]:
                # 开始差异化路由。先一个一个路由，再两个两到主，三个...
                for sendnum in range(1, len(nowRouting)):
                    possible_routing_list = self.possible_routing(len(nowRouting), sendnum)
                    for ri in possible_routing_list:
                        newRouting = ri
                        newLoss, _ = self.predict(awareName, needfeat_all,
                                                  newRouting, nowVgap_dict, now_time, start_time, warehouse_name)
                        if newLoss < 1.0:
                            newLoss = 0.0
                        if newLoss <= tolerable_loss:  # 满足损失要求，结束路由规则
                            finalRouting = self.get_final_routing(origvgap, newRouting)
                            return beginLoss, newLoss, newRouting, finalRouting
                # 到了这里，表示只能所有路由到主了，但是因为是可扩展的，所以还是可以再把没有版本差的发送到备
                newRouting = [1, 1, 1, 1]
                newLoss = nowLoss
                finalRouting = self.get_final_routing(origvgap, newRouting)
                return beginLoss, newLoss, newRouting, finalRouting
            # 到了这里，表示容忍度不满足且不是可扩展的度量级别，则转发到主，结束路由
            newRouting = [1, 1, 1, 1]
            newLoss = nowLoss
            finalRouting = newRouting
            return beginLoss, newLoss, newRouting, finalRouting



# 一下过程有些比较慢，直接一次写入，然后存回文件
# 对warehouse_name进行编码，对时间进行提取，attr扩展转换，动态特征分开。
# 对所有列进行归一化（除warehouse_name、recordid、user_code、
def data_preprocessing():
    sample_data_colums = ['vtime_park_count', 'vtime_loading_time',
                          'vtime_start_time', 'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_finish_time',
                          'vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info', 'vtime_tablea',
                          'vtime_accuqueuedata',
                          'vnum_park_count', 'vnum_loading_time',
                          'vnum_start_time', 'vnum_entry_time', 'vnum_entry_whouse_time', 'vnum_finish_time',
                          'vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info', 'vnum_tablea',
                          'vnum_accuqueuedata',
                          'now_time', 'recordid', 'user_code', 'start_time', 'warehouse_name',
                          'park_count_p', 'park_count_s',
                          'status1_p', 'status1_s', 'status2_p', 'status2_s', 'status3_p', 'status3_s',
                          'predict_p', 'predict_s', 'error', 'gap_time', 'gap_time_second']
    # mydata = pd.read_csv('/root/dataset_accu/collect/sample_data.csv', header=None, names=sample_data_colums, nrows=15000)
    # raw_mydata = pd.read_csv('/root/dataset_accu/collect/sample_data.csv', header=None, names=sample_data_colums)
    mydata = pd.read_csv('/root/dataset_accu/collect/sample_data.csv', header=None, names=sample_data_colums)

    # grouped = raw_mydata.groupby('gap_time', as_index=False)   # 按分钟vgap排序
    # sorted_df = grouped.apply(lambda x: x.sort_values('error'))
    # # mydata = sorted_df.groupby('gap_time').apply(lambda x: x.iloc[len(x)//3: 2*(len(x)//3)])   #中间三分之一
    # # mydata = sorted_df.groupby('gap_time').apply(lambda x: x.iloc[len(x)//4: 3*(len(x)//4)])   #中间二分之一
    # mydata = sorted_df.groupby('gap_time').apply(lambda x: x.iloc[len(x) // 7 * 2: len(x) // 7 * 6])  # 七份，取中间二三四五六


    # *** 时间提取 now_time和start_time
    mydata['now_time_ts'] = pd.to_datetime(mydata['now_time'])
    mydata['now_day'] = mydata['now_time_ts'].dt.day
    mydata['now_hour'] = mydata['now_time_ts'].dt.hour
    mydata['now_minute'] = mydata['now_time_ts'].dt.minute
    mydata['now_second'] = mydata['now_time_ts'].dt.second
    mydata['start_time_ts'] = pd.to_datetime(mydata['start_time'])
    mydata['start_day'] = mydata['start_time_ts'].dt.day
    mydata['start_hour'] = mydata['start_time_ts'].dt.hour
    mydata['start_minute'] = mydata['start_time_ts'].dt.minute
    mydata['start_second'] = mydata['start_time_ts'].dt.second
    # *** attr分解扩展转换( 应该是自动生成转换的，但是这里手动写快一些)
    # vtime_park_count,vnum_park_count 这两个不动
    # status1：start_time、entry_time、loading_time
    # status2：entry_time、entry_whouse_time、loading_time
    # status3：entry_whouse_time、finish_time、loading_time
    mydata['Q1_1_time'] = mydata['vtime_start_time']
    mydata['Q1_2_time'] = mydata['vtime_entry_time']
    mydata['Q1_3_time'] = mydata['vtime_loading_time']
    mydata['Q1_1_num'] = mydata['vnum_start_time']
    mydata['Q1_2_num'] = mydata['vnum_entry_time']
    mydata['Q1_3_num'] = mydata['vnum_loading_time']
    mydata['Q2_1_time'] = mydata['vtime_entry_time']
    mydata['Q2_2_time'] = mydata['vtime_entry_whouse_time']
    mydata['Q2_3_time'] = mydata['vtime_loading_time']
    mydata['Q2_1_num'] = mydata['vnum_entry_time']
    mydata['Q2_2_num'] = mydata['vnum_entry_whouse_time']
    mydata['Q2_3_num'] = mydata['vnum_loading_time']
    mydata['Q3_1_time'] = mydata['vtime_entry_whouse_time']
    mydata['Q3_2_time'] = mydata['vtime_finish_time']
    mydata['Q3_3_time'] = mydata['vtime_loading_time']
    mydata['Q3_1_num'] = mydata['vnum_entry_whouse_time']
    mydata['Q3_2_num'] = mydata['vnum_finish_time']
    mydata['Q3_3_num'] = mydata['vnum_loading_time']
    # 所以，最后扩展的attr为[vtime_park_count, vnum_park_count, 以及上面的Q1-Q3]
    # *** 把动态特征扩展开，注意前两个才是十位的list，第三个状态就是一个值
    def split_list1(datalist):
        d= ast.literal_eval(datalist)
        return pd.Series([d[0], d[1]])
    def split_list2(datalist):
        d= ast.literal_eval(datalist)
        return pd.Series([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]])

    mydata[['status1_p_1', 'status1_p_2']] \
        = mydata['status1_p'].apply(lambda x: split_list1(x))
    mydata[['status1_s_1', 'status1_s_2']] \
        = mydata['status1_s'].apply(lambda x: split_list1(x))

    mydata[['status2_p_1', 'status2_p_2', 'status2_p_3', 'status2_p_4', 'status2_p_5', 'status2_p_6'
        , 'status2_p_7', 'status2_p_8', 'status2_p_9', 'status2_p_10']] \
        = mydata['status2_p'].apply(lambda x: split_list2(x))
    mydata[['status2_s_1', 'status2_s_2', 'status2_s_3', 'status2_s_4', 'status2_s_5', 'status2_s_6'
        , 'status2_s_7', 'status2_s_8', 'status2_s_9', 'status2_s_10']] \
        = mydata['status2_s'].apply(lambda x: split_list2(x))

    # 以上处理后的列有（原基础后接下面的列名）
    # 'now_time_ts', 'now_day', 'now_hour', 'now_minute', 'now_second'    # 访问时间now_time的解析
    # 'start_time_ts', 'start_day', 'start_hour', 'start_minute', 'start_second'    # 排队时间start_time的解析
    # 'Q1_1_time', 'Q1_2_time', 'Q1_3_time', 'Q1_1_num', 'Q1_2_num', 'Q1_3_num',     # attr级别的扩展
    # 'Q2_1_time', 'Q2_2_time', 'Q2_3_time', 'Q2_1_num', 'Q2_2_num', 'Q2_3_num',
    # 'Q3_1_time', 'Q3_2_time', 'Q3_3_time', 'Q3_1_num', 'Q3_2_num', 'Q3_3_num'
    # status1_p_1,status1_p_2,status1_p_3,status1_p_4,status1_p_5,     # 状态1、2十位值分别输入，分主备
    # status1_p_6,status1_p_7,status1_p_8,status1_p_9,status1_p_10,
    # status1_s_1,status1_s_2,status1_s_3,status1_s_4,status1_s_5,
    # status1_s_6,status1_s_7,status1_s_8,status1_s_9,status1_s_10,
    # status2_p_1,status2_p_2,status2_p_3,status2_p_4,status2_p_5,
    # status2_p_6,status2_p_7,status2_p_8,status2_p_9,status2_p_10,
    # status2_s_1,status2_s_2,status2_s_3,status2_s_4,status2_s_5,
    # status2_s_6,status2_s_7,status2_s_8,status2_s_9,status2_s_10

  # 只会用到的列被保存 -----------（最后得到的文件列名是基于此，在最后的‘error’前加warehouse_name的扩展）
    new_mydata = mydata[encoder_colums]
    ###  对数值型特征进行 Min-Max 归一化，中文分类进行OneHot编码
    # 每个列建立单独的归一化处理器，因为后期不知道那些属性会被用到。这你分别单独处理
    x_all = new_mydata.drop('error', axis=1)
    y = new_mydata['error']
    all_warehouse_categories = pd.read_excel('/root/dataset_accu/warehouse_info.xlsx', usecols=['warehouse_name'])
    # 定义归一化处理器和编码器
    scalers = {}
    encoders = {}
    for col in x_all.columns:
        if x_all[col].dtype in ['int64', 'float64']:  # 数值型特征
            scalers[col] = MinMaxScaler()
        else:  # 分类特征
            encoders[col] = OneHotEncoder()
            # 仅对指定列使用独热编码器
            if col == 'warehouse_name':
                encoders[col].fit(all_warehouse_categories)

    # 训练并保存归一化处理器和编码器
    for col, scaler in scalers.items():
        scaler.fit(x_all[[col]])
        x_all[[col]] = scaler.transform(x_all[[col]])
        with open(f'data_pkl/{col}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    for col, encoder in encoders.items():
        with open(f'data_pkl/{col}_encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)

    # 将归一化后的特征和目标变量连接起来
    for col, encoder in encoders.items():
        # 对分类特征进行编码
        encoded_data = encoder.transform(x_all[[col]])
        # 获取编码后的列名
        feature_names = encoder.get_feature_names([col])
        # 将编码后的特征与原始特征合并
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=feature_names) #将独热编码后的稀疏矩阵转换为二维数组，以便将其转换为 DataFrame
        x_all = x_all.drop(col, axis=1)
        x_all = pd.concat([x_all, encoded_df], axis=1)

    # 合并特征和目标变量
    new_mydata_normalized = pd.concat([x_all, y], axis=1)
    new_mydata_normalized.to_csv('/root/dataset_accu/collect/new_sample_data.csv', index=False)


# 同获取特征平均sql数。虽然在colltec4.py中也收集了sql，但是这里用不上。只需要统计数文件就好
def feature_neednum():
    # 后面两位是list，表示的获取park_count、status1、status2、status的个数
    data_colums = ['now_time', 'start_time', 'exesql_num_p', 'exesql_num_s']
    df = pd.read_csv('/root/dataset_accu/collect/expt_sql_num0.csv', header=None, names=data_colums)

    # 可加入特征. 第一组为自带的，第二组是从备库获取的，第三组是从主库获取的
    candidate_features0 = ['now_time', 'start_time', 'warehouse_name']
    candidate_features_s = ['park_count_s', 'status1_s', 'status2_s', 'status3_s']
    candidate_features_p = ['park_count_p', 'status1_p', 'status2_p', 'status3_p']

    # 统计二三组的平均00所需获取sql数
    df['exesql_num_p'] = df['exesql_num_p'].apply(ast.literal_eval)
    df['exesql_num_s'] = df['exesql_num_s'].apply(ast.literal_eval)

    # 使用apply和lambda函数将列表中的值展开到新的列中
    expanded_df_p = df['exesql_num_p'].apply(pd.Series)
    expanded_df_s = df['exesql_num_s'].apply(pd.Series)

    # 计算均值
    column_means_p = expanded_df_p.mean().tolist()
    column_means_s = expanded_df_s.mean().tolist()

    # kv对应并排序
    k = candidate_features_s + candidate_features_p
    v = column_means_s + column_means_p

    feature_mean_dict = dict(zip(k, v))
    sorted_feature_mean = dict(sorted(feature_mean_dict.items(), key=lambda x: x[1]))
    sort_k = list(sorted_feature_mean.keys())
    sort_v = list(sorted_feature_mean.values())
    print(sort_k)
    print(sort_v)
    allfeature_sort_k = candidate_features0 + sort_k

    # 输出
    # k = ['park_count_s', 'park_count_p', 'status3_s', 'status3_p',
    #      'status2_s', 'status2_p', 'status1_p', 'status1_s']
    # v = [1.0, 1.0, 1.705336426914153, 1.7616350484509349,
    #      6.89480687866794, 7.179575542513989, 13.786611164187253, 16.111164187252626]


if __name__ == "__main__":
    # 对样本数据进行处理--id、扩展
    # data_preprocessing()
    # print("数据预处理完成")

    # 对特征获取所需平均sql数计算     !!!!!!
    # feature_neednum()   #这里的结果手动写到 parameter.py里的feature_add

    # 构建 仅新鲜度的模型 和 k11 模型
    # sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')   # 这个预处理后的文件有列名，就是上面对应的
    # aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    # aware_service.build_awaremodel(11)


    sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')   # 这个预处理后的文件有列名，就是上面对应的
    aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    for kii in range(1, 11):
        print(f"开始增加 {kii} 个特征")
        aware_service.build_awaremodel(kii)

    # 制定属性列名,会保存结果到结果文件
    sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')  # 这个预处理后的文件有列名，就是上面对应的
    aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    base_level = [aware_service.db_t, aware_service.tab_t,
                  aware_service.attr_t, aware_service.attrextend_t,
                  aware_service.all_t]
    base_level_name = ['db', 'tab', 'attr', 'attrextend', 'all']
    k11 = ['start_time', 'warehouse_name', 'now_time', 'park_count_s', 'park_count_p',
           'status3_s', 'status3_p', 'status2_s', 'status2_p', 'status1_p', 'status1_s']
    k3 = ['start_time', 'warehouse_name', 'now_time']

    for i in range(1, len(k11)):
    # for i in [3]:
        colname_all = k11[:i]  #不含右限
        print(f"#######加入特征个数 {i}")
        replaced_colname = list()
        for ki in colname_all:
            replaced_colname.extend(aware_service.replace_cloname(ki))
        # print(replaced_colname)
        for leveli in range(len(base_level)):
            colname_all_copy = colname_all.copy()  # 创建一个新的列表，而不是修改现有列表
            allfeat = base_level[leveli] + replaced_colname
            # print(allfeat)
            x_train, x_test, y_train, y_test = aware_service.get_train_data(allfeat)
            aware_m, eval_r = aware_service.choose_model(x_train, x_test, y_train, y_test)

            modelname = f"aware_{base_level_name[leveli]}_t_add{i}"
            vgapname = f"{base_level_name[leveli]}_t"
            colname_all_copy.insert(0, vgapname)
            modelinfo = [[modelname, colname_all_copy] + eval_r]
            with open('/root/dataset_accu/collect/eval_result.csv', 'a+') as file:
                writer = csv.writer(file)
                writer.writerows(modelinfo)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            midelpath = f"{modelname}.pkl"
            model_path = os.path.join(script_dir, 'model_pkl', midelpath)
            with open(model_path, 'wb') as file:
                pickle.dump(aware_m, file)

    # # # 指定特征，不会保存，只输出看结果
    # k11 = ['start_time', 'warehouse_name', 'now_time', 'park_count_s', 'park_count_p',
    #        'status3_s', 'status3_p', 'status1_s', 'status2_s', 'status1_p', 'status2_p']
    # k3 = ['start_time', 'warehouse_name', 'now_time']
    # kall = [k3]
    #
    # sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')  # 这个预处理后的文件有列名，就是上面对应的
    # aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    # base_level = [aware_service.db_t, aware_service.tab_t,
    #               aware_service.attr_t, aware_service.attrextend_t,
    #               aware_service.all_t]
    # for k in kall:
    #     replaced_colname = list()
    #     for ki in k:
    #         replaced_colname.extend(aware_service.replace_cloname(ki))
    #     allfeat = base_level[2] + replaced_colname
    #     x_train, x_test, y_train, y_test = aware_service.get_train_data(allfeat)
    #     _, eval_r = aware_service.choose_model(x_train, x_test, y_train, y_test)
    #     print(eval_r)





