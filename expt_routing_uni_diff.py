import csv

import pandas as pd
from datetime import datetime, timedelta

import pymysql
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wait.aware_service import AccuAwareService, data_preprocessing
from Wait.create_data import warehousedata_to_mysql, cargodata_to_mysql
from Wait.parameter import feature_vgap_tn, feature_vgap_t, feature_add, feature_vgap_n, attr_version
from Wait.queue_status import update_data, cal_request_trucks, predict_wait_time
from Wait.version_txn import create_versiontable, get_vgap, version_syn

### 使用构建好的模型，接着之前负载的场景参数，开始比对路由
## 收集不同容忍值下，到备库的差U型那时间数，其中的loss占比
from Wait.workload import QueueWorkload2


# 无关项再表中的初始时间值。注意主备都要修改，还有版本表
def update_start_intcol_inversion(time_str):
    for ip in ['106.75.233.244', '106.75.244.49']:
        db = pymysql.connect(host=ip, user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()

        # 无关项
        sql = "UPDATE queuing_info SET qt = %s WHERE recordid = 10;"
        cursor.execute(sql, (time_str, ))
        sql = "UPDATE warehouse_info SET wt = %s;"
        cursor.execute(sql, (time_str,))
        sql = "UPDATE cargo_info SET ct = %s;"
        cursor.execute(sql, (time_str,))
        # 静态特征的time版本表
        sql = "UPDATE accuqueueversion.vtime_park_count SET version = %s;"
        cursor.execute(sql, (time_str,))
        sql = "UPDATE accuqueueversion.vtime_loading_time SET version = %s;"
        cursor.execute(sql, (time_str,))

        db.commit()
        cursor.close()
        db.close()


# 为limit-time路由策略准备的，因为这个只涉及主备库最新事务提交时间，传入参数是个dict
# 这里另外把表级别也记录上，以免有效用
def get_vgap_dbtab(vgap_time):
    # 在记录版本号的时候没有额外标识db和tab等，用的是名字，这里直接固定写，改前面改动太多
    vgap_db = vgap_time['vtime_accuqueuedata']
    vgap_tab = [vgap_time['vtime_queuing_info'], vgap_time['vtime_cargo_info'], vgap_time['vtime_warehouse_info'],
                vgap_time['vtime_tablea']]
    return vgap_db, vgap_tab

def workload_file_model():
    # 1。负载配置文件生成
    gapnum = 1500   # 虽然实验是1000，但是收集的时候是1500.这里指在后面有比例随机的，所以还是不变
    queue_wk = QueueWorkload2(1, gapnum)
    # 2.对比模型保存为pkl
    # data_preprocessing()  # 编码器
    sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')   # 这个预处理后的文件有列名，就是上面对应的
    aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    aware_service.build_awaremodel(3)  # 这里会生成仅新鲜度的，以及add3的各级别下的模型

def syn_backup(backup_time):
    queue_wk.queuing_syn()  # 包含对库下的三个表的同步，使用SQL语句同步
    version_syn()
    return backup_time

def nextday_insert_startdata(update_time):
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()
    sql = "SELECT warehouseid FROM warehouse_info WHERE park_count < 6 limit 3;"
    cursor.execute(sql)
    r = cursor.fetchall()
    cursor.close()
    db.close()

    queue_wk.create_queuedata([r[0][0]], 5, update_time.strftime('%Y-%m-%d %H:%M:%S'), 10)  # 仓库编号，车辆数，到达时间base，时间间隔
    queue_wk.create_queuedata([r[1][0]], 3, '', 10)  # 时间参数为空，会获取last记录的时间的start_time作为base
    queue_wk.create_queuedata([r[0][0]], 4, '', 5)
    queue_wk.create_queuedata([r[2][0]], 6, '', 5)
    queue_wk.create_queuedata([r[1][0]], 4, '', 5)
    queue_wk.create_queuedata([r[2][0]], 2, '', 5)
    queue_wk.create_queuedata([r[0][0]], 2, '', 5)
    add_time = 5 * 10 + 3 * 10 + 4 * 5 + 6 * 5 + 4 * 5 + 2 * 5 + 2 * 5
    update_time += timedelta(seconds=add_time)

    return update_time


if __name__ == "__main__":
    # 0. 这步单独执行，一次就行，生成需要的对比模型和负载配置数据
    # workload_file_model()
    # 1.初始数据环境（将数据库清空）
    warehousedata_to_mysql()
    cargodata_to_mysql()
    start_time_str = datetime.strptime('2021-01-15 05:59:50', '%Y-%m-%d %H:%M:%S')
    create_versiontable()
    update_start_intcol_inversion(start_time_str)  # 将相关表中的无关项初始化换为测试的起始时间，还有版本表初始时间

    # 2.接着收集数据后的状态开始测试 （目前不是直接接，而是从一个点开始）
    gapnum = 1000
    queue_wk = QueueWorkload2(0, gapnum)  # 一个空queuing_info表，参数1不会变动负载参数文件，参数2是gap时间点个数
    # queue_wk.fast_i = 2112   # 以下这些参数还是不设置接着样本收集之后的，因为gap次数在生成某些配置文件的时候有占比
    # queue_wk.slow_i = 569    # 此时若是跳索引号的话，会导致超过index
    # queue_wk.veryslow_i = 203
    # queue_wk.wh_i = 943
    # queue_wk.car_i = 1471
    # queue_wk.type_i = 710
    # queue_wk.s_type_i = 100
    # queue_wk.s_num_i = 100
    # queue_wk.s_time_i = 144
    # queue_wk.s_loading_i = 67
    # queue_wk.s_park_i = 77
    # queue_wk.p_time = 746
    # queue_wk.ta_num_i = 1500
    # queue_wk.ta_time_i = 20632
    # queue_wk.tc_num_i = 4500
    # queue_wk.tc_time_i = 50912

    # 3，加载初始数据
    now_time = start_time_str
    # 一下初始数据随机更改，需要和data_collect4.py收集数据时使用的不一样
    queue_wk.create_queuedata([1], 5, now_time.strftime('%Y-%m-%d %H:%M:%S'), 10)
    queue_wk.create_queuedata([2], 3, '', 20)
    queue_wk.create_queuedata([1], 4, '', 5)
    queue_wk.create_queuedata([5], 6, '', 8)
    queue_wk.create_queuedata([2], 4, '', 5)
    queue_wk.create_queuedata([5], 2, '', 5)
    queue_wk.create_queuedata([1], 2, '', 5)
    add_time = 5 * 10 + 3 * 20 + 4 * 5 + 6 * 8 + 4 * 5 + 2 * 5 + 2 * 5
    now_time += timedelta(seconds=add_time)
    update_data('106.75.233.244', now_time)  # 更新主库状态
    backup_time = syn_backup(now_time)  # 同步备库

    # 4.准备训练好的感知模型
    sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')
    aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    # 注意：这里提前在AccuAwareService类中读入之前导入的模型结果文件数据，便于根据模型名获取对应的属性列
    aware_service.get_awarefeat_info()
    # 注意，这里还需将对比的模型提前加载好。 提前确认传入的awaremodel name是否存在
    compare_model = ['aware_attrextend_t_add3']
    aware_service.get_awaremodel_info(compare_model)
    # 注意，加载归一化和独热编码器 （这里全部加载了，因为有些列名有改动过）
    aware_service.get_data_pkl()

    with open('/root/dataset_accu/collect/expt_uni_diff.csv', 'w+') as limitfile:
        pass
    # 5.同一个模型，不同路由策略
    tolerable_loss = [0, 5, 10, 15, 20, 25, 30, 45, 60, 120, 240]  # 尽量写全，这个是得慢慢跑负载流程的

    user_request_all = list()
    for i in range(1, gapnum+1):  # ***** 对每个vgap
        print(f"*****vgap轮数：{i}*****")
        now_time = queue_wk.dynmaicload(now_time, i)  # 插入新车
        update_data('106.75.233.244', now_time)  # 更新主库

        # 5.1) 版本信息
        vgap_time, vgap_num = get_vgap()
        #  recordid, user_code, cargo_name, warehouse_name, start_time
        now_request_trucks = cal_request_trucks()
        user_num = len(now_request_trucks)
        print(user_num)
        # 添加下面这段操作是因为，当天已经没有新车来了，旧车也全部进入了，要进入下一天了。不再收集数据，直接跳到同步。然后进入下一个gap
        if user_num == 0 and now_time.hour >= 13:  # 这个对比的hour和负载设置定的更新频率划分的最后时间相同
            next_day = now_time + timedelta(days=1)
            now_time = next_day.replace(hour=6, minute=0, second=0)
            # 新一天开始增开始数据。同步到备之后，回到main函数，开始新一天的第一次gap
            # print("进入下一天 #################")
            now_time = nextday_insert_startdata(now_time)
            update_data('106.75.233.244', now_time)

            backup_time = syn_backup(now_time)  # 同步备库，进入下轮收集
            break

        for user_i in now_request_trucks:   # ***** 对每个user
            recordid = user_i[0]
            user_code = user_i[1]
            warehouse_name = user_i[3]
            start_time = user_i[4]

            # 1）全路由到主库的计算结果，真实值
            p_cal, _, _, _ = predict_wait_time([1, 1, 1, 1], now_time, start_time, warehouse_name)
            # 2） 路由到备库
            s_cal, _, _, _ = predict_wait_time([0, 0, 0, 0], now_time, start_time, warehouse_name)


            # 结果收集记录
            # 请求时间、用户标识1、2、真实值、
            result = []
            result.append(now_time)  # 请求时间
            result.append(recordid)  # 标识用户
            result.append(user_code)  # 标识用户
            result.append(p_cal)  # 到主库的就是真实结果
            result.append(s_cal)  # 全到备
            result.append(abs(p_cal-s_cal))  # 真实损失

            # ***** 对每个容忍度.
            # 记录-- 当前版本差异下预测的loss
            # 记录-- 每个容忍值下都记录：设定的loss、diff方法路由结果真实的loss，diff确定方法后预测loss，diff路由结果、uni路由结果
            awaremodel_name = compare_model[0]
            # 只传入会使用到的度量方式的信息
            if '_tn_' in awaremodel_name:
                nowVgap = {**vgap_time, **vgap_num}
            elif '_n_' in awaremodel_name:
                nowVgap = vgap_num
            else:
                nowVgap = vgap_time
            for tloss in tolerable_loss:
                awarebeginLoss, newLoss, newRouting, finalRouting = \
                    aware_service.routing(tloss, awaremodel_name, nowVgap, now_time, start_time, warehouse_name)

                # 这个记录一次就行了
                if tloss == tolerable_loss[0]:
                    result.append(awarebeginLoss)

                # 感知结果  （预测的结果，根据确定路由实时计算的结果）
                aware_predict = newLoss  # 感知服务路由结果 (模型预测的）
                aware_cal, _, _, _ = predict_wait_time(finalRouting, now_time, start_time, warehouse_name)

                result.append(tloss)  # 此时的容忍度
                result.append(aware_cal)  # aware真实结果--diff
                result.append(aware_predict)  # aware预测结果--diff
                result.append(finalRouting)  # 真实路由路径，因为到主的vgap0，可以到备--diff

                if awarebeginLoss <= tloss:   # ----uni
                    result.append([0, 0, 0, 0])
                else:
                    result.append([1, 1, 1, 1])

            with open('/root/dataset_accu/collect/expt_uni_diff.csv', 'a+') as file:
                writer = csv.writer(file)
                writer.writerow(result)

        backup_time = syn_backup(now_time)  # 同步备库，进入下轮收集
        user_request_all.append(len(now_request_trucks))  # 记录次轮gap收集到的数据条数

    queue_wk.view_classdata()
    # 每次gap可请求数记录
    with open("/root/dataset_accu/collect/expt_user_request_times.txt", 'w') as file:
        file.write(str(user_request_all))




