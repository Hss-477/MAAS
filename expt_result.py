import csv
import pickle

import pandas as pd
from datetime import datetime, timedelta

import pymysql
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wait.aware_service import AccuAwareService
from Wait.create_data import warehousedata_to_mysql, cargodata_to_mysql
from Wait.parameter import feature_vgap_tn, feature_vgap_t, feature_add, feature_vgap_n
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

def syn_backup(backup_time):
    queue_wk.queuing_syn()  # 包含对库下的三个表的同步，使用SQL语句同步。新补充tavlea的同步
    version_syn()
    return backup_time


# 为limit-time路由策略准备的，因为这个只涉及主备库最新事务提交时间，传入参数是个dict
# 这里另外把表级别也记录上，以免有效用
def get_vgap_dbtab(vgap_time):
    # 在记录版本号的时候没有额外标识db和tab等，用的是名字，这里直接固定写，改前面改动太多
    vgap_db = vgap_time['vtime_accuqueuedata']
    vgap_tab = [vgap_time['vtime_queuing_info'], vgap_time['vtime_cargo_info'], vgap_time['vtime_warehouse_info'],
                vgap_time['vtime_tablea']]
    return vgap_db, vgap_tab


# 新一天开始，手动添加数据
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
    queue_wk.create_queuedata([r[0][0]], 4, '', 8)
    queue_wk.create_queuedata([r[2][0]], 6, '', 20)
    queue_wk.create_queuedata([r[1][0]], 4, '', 5)
    queue_wk.create_queuedata([r[2][0]], 2, '', 5)
    queue_wk.create_queuedata([r[0][0]], 2, '', 5)
    add_time = 5 * 10 + 3 * 10 + 4 * 8 + 6 * 20 + 4 * 5 + 2 * 5 + 2 * 5
    update_time += timedelta(seconds=add_time)

    return update_time


# 使用regret方法路由，返回路由结果，以及此路由真实路由后得到的预测值
def regret_routing(ntime, stime, wname, alltoprimary):
    base = alltoprimary
    regert1, _, _, _, _, _ = predict_wait_time([1, 0, 0, 0], ntime, stime, wname)
    regert2, _, _, _, _, _ = predict_wait_time([0, 1, 0, 0], ntime, stime, wname)
    regert3, _, _, _, _, _ = predict_wait_time([0, 0, 1, 0], ntime, stime, wname)
    regert4, _, _, _, _, _ = predict_wait_time([0, 0, 0, 1], ntime, stime, wname)

    # 差值越小，说明影响越大。（因为仅这个值取最新的，预测结果越接近真实的）
    regret_all = [abs(base-regert1), abs(base-regert2), abs(base-regert3), abs(base-regert4)]
    regret_result = list()

    # 找出导致误差最大的两个值，将其路由到主库
    # 相当于保证影响最大的特征，使用最新数据计算
    for imax in range(1, 4):
        sorted_list = sorted(regret_all)  # 从小到大
        max_values = sorted_list[:imax]
        max_indexes = [regret_all.index(value) for value in max_values]
        nowrouting = [0, 0, 0, 0]
        for ri in max_indexes:
            nowrouting[ri] = 1
        regertnow, _, _, _, _, _ = predict_wait_time(nowrouting, ntime, stime, wname)
        regret_result.append(nowrouting)
        regret_result.append(regertnow)

    # 返回的是每个更新数选1,2,3时，得到的路由结果和这个路由真实会产生的loss
    # 所以这是有6个数据的list
    return regret_result


# 因为之前收集样本的输出没有保存最后的时间，所以这里先用查询到的最大时间来替代
def train_maxtime():
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()

    sql = "SELECT MAX(start_time), MAX(entry_time), MAX(entry_whouse_time), MAX(finish_time) FROM queuing_info;"
    cursor.execute(sql)
    r = cursor.fetchone()
    result = [r[0], r[1], r[2], r[3]]
    now = max(result)
    cursor.close()
    db.close()

    return now


# !!!!!!! expt_routing_uni_diff.py文件是可以运行的，他们有覆盖，所以这里用写到一起
if __name__ == "__main__":
    # 1.初始数据环境（将数据库清空）
    # warehousedata_to_mysql()
    # cargodata_to_mysql()
    # start_time_str = datetime.strptime('2021-04-01 05:59:50', '%Y-%m-%d %H:%M:%S')
    # create_versiontable()
    # update_start_intcol_inversion(start_time_str)  # 这里再加一个函数，将相关表中的无关项初始化换为测试的起始时间，还有版本表初始时间
    ############## 以上暂时不用了，手动恢复之前的状态

    # 2.接着收集数据后的状态开始测试 （目前不是直接接，而是从一个点开始）
    gapnum = 800  #！！！！！！ 这个gap传进去是没用的，这个设置是要接train的，
    queue_wk = QueueWorkload2(0, gapnum)  # 一个空queuing_info表，参数1不会变动负载参数文件和数据库数据，参数2是gap时间点个数
    queue_wk.fast_i = 590
    queue_wk.slow_i = 150
    queue_wk.veryslow_i = 41
    queue_wk.wh_i = 258
    queue_wk.car_i = 381
    queue_wk.type_i = 192
    queue_wk.s_type_i = 28
    queue_wk.s_num_i = 28
    queue_wk.s_time_i = 39
    queue_wk.s_loading_i = 29
    queue_wk.s_park_i = 9
    queue_wk.p_time = 194
    queue_wk.ta_num_i = 400
    queue_wk.ta_time_i = 5265
    queue_wk.tc_num_i = 1200
    queue_wk.tc_time_i = 8195

    # 3，加载初始数据
    # now_time = start_time_str
    # # 一下初始数据随机更改，需要和data_collect4.py收集数据时使用的不一样
    # queue_wk.create_queuedata([1], 5, now_time.strftime('%Y-%m-%d %H:%M:%S'), 10)  # 仓库编号，车辆数，到达时间base，时间间隔
    # queue_wk.create_queuedata([2], 3, '', 20)  # 时间参数为空，会获取last记录的时间的start_time作为base
    # queue_wk.create_queuedata([1], 4, '', 5)
    # queue_wk.create_queuedata([5], 6, '', 8)
    # queue_wk.create_queuedata([2], 4, '', 5)
    # queue_wk.create_queuedata([5], 2, '', 5)
    # queue_wk.create_queuedata([1], 2, '', 5)
    # add_time = 5 * 10 + 3 * 20 + 4 * 5 + 6 * 8 + 4 * 5 + 2 * 5 + 2 * 5
    # now_time += timedelta(seconds=add_time)
    # update_data('106.75.233.244', now_time)  # 更新主库状态
    # backup_time = syn_backup(now_time)  # 同步备库
    ############  这个地方还是应该改为训练数据时的负载参数，不然 感觉走向不对
    # 因为之前并没与返回这个，现在这里使用查表，各时间列最大的作为当前时间
    # 后续不需要执行什么操作，直接开始调动态更新的函数，形成版本差收集数据即可
    # now_time = train_maxtime()
    now_time = datetime.strptime('2021-01-19 10:52:51', '%Y-%m-%d %H:%M:%S')


    # 4.准备训练好的感知模型
    sample_data = pd.read_csv('/root/dataset_accu/collect/new_sample_data.csv')
    aware_service = AccuAwareService(feature_vgap_tn, feature_vgap_t, feature_vgap_n, feature_add, sample_data)
    # 注意：这里提前在AccuAwareService类中读入之前导入的模型结果文件数据，便于根据模型名获取对应的属性列
    aware_service.get_awarefeat_info()
    # 注意，这里还需将对比的模型提前加载好。 提前确认传入的awaremodel name是否存在
    compare_model_nuidiff = ['aware_attrextend_t_add3']
    compare_model = ['aware_attrextend_t', 'aware_attrextend_t_add3']
    # aware_service.get_awaremodel_info(compare_model_nuidiff)  # 这里注释掉，因为包含在下面的
    aware_service.get_awaremodel_info(compare_model)
    # 注意，加载归一化和独热编码器 （这里全部加载了，因为有些列名有改动过）
    aware_service.get_data_pkl()


    # 5.每个gap时间点收集路由结果数据
    # 对每个vgap时
    # 每个用户
    # 不同容忍度
    # 不同感知模型    结果写入对应文件
    # 存放文件开启：一个limit-time、多个aware-mddel（同文件下，同容忍、不同模型）

    tolerable_loss = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 120, 240]  # 尽量写全，这个是得慢慢跑负载流程的
    tolerable_time = [5, 10, 15, 20, 25, 30, 45, 60, 120, 240]  # 不用写太多，后期也能根据保存的版本信息计算

    # 第一次创建一堆空文件放数据，后面使用a+模型插入数据
    # 对应列： 发送时间，排队时间，sql句子
    with open('/root/dataset_accu/collect/expt_sql_p.csv', 'w+') as sqlpfile:
        pass   # 分为两个原因是主备库上执行的不一样。虽然后面跑tps只用主的，但这里记录下来，备用
    with open('/root/dataset_accu/collect/expt_sql_s.csv', 'w+') as sqlsfile:
        pass
    with open('/root/dataset_accu/collect/expt_sql_num.csv', 'w+') as sqlnumfile:
        pass  # 这个文件用于记录sql主备语句数的记录,另外接着两个标识now_time, 排队时间,
    # 对应列：访问时间，recordid, user_code,真实值、全备、误差, aware第一次预测的结果
    # * 对每个容忍度.
    # 记录-- 当前版本差异下预测的loss
    # 记录-- 每个容忍值下都记录：设定的loss、diff方法路由结果真实的loss，diff确定方法后预测loss，diff路由结果、uni路由结果
    with open('/root/dataset_accu/collect/expt_uni_diff.csv', 'w+') as difffile:
        pass
    # 对应列：库级别time、表级别time、请求时间now_time、recordid、user_code、
    #        真实值p_cal、全备结果s_cal、损失error
    #        针对[容忍时间分别为5, 10, 15, 20, 25, 30, 45, 60, 120分钟]的路由结果，0或1
    with open('/root/dataset_accu/collect/expt_limit_time.csv', 'w+') as limitfile:
        pass
    # 对应有：请求时间，usercode，真实值
    #   针对最大更新选项为[1,2,3]个时，收集：路由结果，此时路由结果真实算出来的排队时间
    with open('/root/dataset_accu/collect/expt_regret.csv', 'w+') as regretfile:
        pass
    # 对应列： 访问时间now_time、recordid、user_code、真实值p_cal、
    #         对每个模型有三列结果：感知真实结果aware_cal、感知预测结果aware_predict、newRouting、查询路由结果finalRouting
    for tloss in tolerable_loss:
        awarePath = f"/root/dataset_accu/collect/expt_aware_{tloss}.csv"
        with open(awarePath, 'w+') as awarefile:
            pass

    user_request_all = list()
    start_gap = 400 + 1
    for i in range(start_gap, start_gap+gapnum):  # ***** 对每个vgap  ！！！！！！这个地方改成train的gap+我们要结果测试的gap
                                  # 包左不包右。第一个是之前收集数据的轮数，第二个累加上要执行的轮数
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
            update_data('106.75.233.244', now_time)
            backup_time = syn_backup(now_time)
            now_time = nextday_insert_startdata(now_time)  # 新一天开始来车
            update_data('106.75.233.244', now_time)
            now_request_trucks = cal_request_trucks()
            user_num = len(now_request_trucks)
            print(user_num)

        for user_i in now_request_trucks:   # ***** 对每个user
            recordid = user_i[0]
            user_code = user_i[1]
            warehouse_name = user_i[3]
            start_time = user_i[4]

            # ！！！！！！！ 主库、备库、限制时间
            # 5.2) 模型 ： direct-p、direct-s、、limit-time（多个限制时间）、aware-model（多个模型，多个容忍值）
            # 5.2.1）全路由到主库的计算结果，真实值
            # 注意下面这个函数增加了两个参数和返回值。参数默认可以不写。为1分别表示：是否保存执行的sql语句、执行sql语句的条数
            p_cal, _, _, _, exesql_num_p, _ = predict_wait_time([1, 1, 1, 1], now_time, start_time, warehouse_name, 1, 1)
            # 5.2.2） 路由到备库
            s_cal, _, _, _, _, exesql_num_s = predict_wait_time([0, 0, 0, 0], now_time, start_time, warehouse_name, 1, 1)
            # 5.2.3） limit-time
            vgap_db, vgap_tab = get_vgap_dbtab(vgap_time)  # limit_time方法需要以db time来判断，这里另外把表级别也记录上
            # 结果收集记录
            result_limit = []
            result_limit.append(vgap_db)  # 库的时间差异
            result_limit.append(vgap_tab)  # 表级别  （四个表放在一个list里）
            result_limit.append(now_time)  # 请求时间
            result_limit.append(recordid)  # 标识用户
            result_limit.append(user_code)  # 标识用户
            result_limit.append(p_cal)  # 到主库的就是真实结果
            result_limit.append(s_cal)  # 全到备
            result_limit.append(abs(p_cal-s_cal))  # 预测损失error
            for tlimit in tolerable_time:
                if vgap_db <= tlimit:  # 比较的都是以 分钟 为单位
                    result_limit.append(0)   # 当前主备库差异可接受，路由到被，标识为0。预测差异为 abs(p_cal-s_cal)
                else:  # 超过时限，标识为1，表示要去主，此时差异为0
                    result_limit.append(1)

            with open('/root/dataset_accu/collect/expt_limit_time.csv', 'a+') as limitfile:
                writer = csv.writer(limitfile)
                writer.writerow(result_limit)
            with open('/root/dataset_accu/collect/expt_sql_num.csv', 'a+') as sqlnumfile:
                result_sqlnum = [now_time, start_time, exesql_num_p, exesql_num_s]
                writer = csv.writer(sqlnumfile)
                writer.writerow(result_sqlnum)


            # ！！！！！！！ regret方法
            result_regret = []
            result_regret.append(now_time)  # 请求时间
            result_regret.append(recordid)  # 标识用户
            result_regret.append(p_cal)  # 到主库的就是真实结果
            r_result = regret_routing(now_time, start_time, warehouse_name, p_cal)
            result_regret.extend(r_result)
            with open('/root/dataset_accu/collect/expt_regret.csv', 'a+') as regretfile:
                writer = csv.writer(regretfile)
                writer.writerow(result_regret)

            # ！！！！！！！ 不同aware模型 （因为差异化路由的实验有重叠，这里混合进来）
            # 5.2.4) ***** 对每个容忍度  （这个是多种方法对比实验的收集）
            result_diff = []
            # 结果收集记录：请求时间、用户标识1、2、真实值、
            result_diff.append(now_time)  # 请求时间
            result_diff.append(recordid)  # 标识用户
            result_diff.append(user_code)  # 标识用户
            result_diff.append(p_cal)  # 到主库的就是真实结果
            result_diff.append(s_cal)  # 全到备
            result_diff.append(abs(p_cal - s_cal))  # 真实损失
            for tloss in tolerable_loss:
                # 结果收集记录
                result_aware = []
                result_aware.append(now_time)  # 请求时间
                result_aware.append(recordid)  # 标识用户
                result_aware.append(user_code)  # 标识用户
                result_aware.append(p_cal)  # 到主库的就是真实结果

                for awaremodel_name in compare_model:  # 每个模型，有三列数据，aware真实、aware预测、路由路径
                    # 只传入会使用到的度量方式的信息
                    if '_tn_' in awaremodel_name:
                        nowVgap = {**vgap_time, **vgap_num}
                    elif '_n_' in awaremodel_name:
                        nowVgap = vgap_num
                    else:
                        nowVgap = vgap_time

                    awarebeginLoss, newLoss, newRouting, finalRouting= \
                        aware_service.routing(tloss, awaremodel_name, nowVgap, now_time, start_time, warehouse_name)
                    # 感知结果  （预测的结果，根据确定路由实时计算的结果）
                    aware_predict = newLoss  # 感知服务路由结果 (模型预测的）
                    aware_cal, _, _, _, _, _ = predict_wait_time(finalRouting, now_time, start_time, warehouse_name)

                    result_aware.append(aware_cal)  # aware真实结果
                    result_aware.append(aware_predict)  # aware预测结果
                    result_aware.append(newRouting)   # 原本判断的路由结果，但不是最后使用的
                    result_aware.append(finalRouting)  # 真实路由路径，因为到主的vgap0，可以到备

                    # ！！！！！！！ 差异化路由对比
                    if awaremodel_name == compare_model_nuidiff[0]:
                        # 这个记录一次就行了
                        if tloss == tolerable_loss[0]:
                            result_diff.append(awarebeginLoss)
                        result_diff.append(tloss)  # 此时的容忍度
                        result_diff.append(aware_cal)  # aware真实结果--diff
                        result_diff.append(aware_predict)  # aware预测结果--diff
                        result_diff.append(finalRouting)  # 真实路由路径，因为到主的vgap0，可以到备--diff
                        if awarebeginLoss <= tloss:  # ----uni
                            result_diff.append([0, 0, 0, 0])
                        else:
                            result_diff.append([1, 1, 1, 1])


                awarePath = f"/root/dataset_accu/collect/expt_aware_{tloss}.csv"
                with open(awarePath, 'a+') as awarefile:
                    writer = csv.writer(awarefile)
                    writer.writerow(result_aware)
            with open('/root/dataset_accu/collect/expt_uni_diff.csv', 'a+') as difffile:
                writer = csv.writer(difffile)
                writer.writerow(result_diff)


        backup_time = syn_backup(now_time)  # 同步备库，进入下轮收集
        user_request_all.append(user_num)

    queue_wk.view_classdata()
    # 每次gap可请求数记录
    with open("/root/dataset_accu/collect/expt_user_request_times_compare.txt", 'w') as file:
        file.write(str(user_request_all))




