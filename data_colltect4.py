import csv
import random
from datetime import datetime, timedelta
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Wait.create_data import warehousedata_to_mysql, cargodata_to_mysql
from Wait.queue_status import update_data, cal_request_trucks, predict_wait_time
from Wait.version_txn import create_versiontable, version_syn, get_vgap
from Wait.workload import QueueWorkload2


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


# 在每次主备之间有状态差异的时候收集数据
def collect_sample():  # 在main中，主库时间为now_time，备库时间是backup_time
    global now_time, backup_time, user_request_all
    # 1)版本差信息time和num， 依次属性、表、库级别
    # 2)特征 now_time、start_time、warehouse_name、cargo_name
    # 3)动态特征,过滤得到排在前的车辆的信息，这里只要货物加载时间信息
    #             status1.队伍前的车辆的加载时间list；   厂外前车等待数和等待时间求和
    #             status2库外停车的车辆的加载时间list；  最多10位
    #             status3库内工作车辆的剩余时间。      
    vgap_time, vgap_num = get_vgap()
    now_request_trucks = cal_request_trucks()  # 目前在排队的车辆，可以发出预测请求
    # print(f"可发送请求数：{len(now_request_trucks)}，具体信息是：{now_request_trucks}")
    # print(now_request_trucks)  #属性分别是：recordid, user_code, cargo_name, warehouse_name, start_time

    # print(f"本次收集间隔时间（分钟）：{(now_time-backup_time).total_seconds()/60}，可访问的者数：{len(now_request_trucks)}")
    # 如果时间在21点后，且可访问车数为0，证明今天工作结束，跳到第二天早6点
    user_num = len(now_request_trucks)
    print(user_num)
    if user_num == 0 and now_time.hour >= 13:  # 这个对比的hour和负载设置定的更新频率划分的最后时间相同
        next_day = now_time + timedelta(days=1)
        now_time = next_day.replace(hour=6, minute=0, second=0)
        # 新一天开始增开始数据。同步到备之后，回到main函数，开始新一天的第一次gap
        # print("进入下一天 *")
        #注意，这里修改修改一下，为了减少0请求访问的产生，此时跳到第二天，主备状态更新这个，下面新加数据是可以访问的
        update_data('106.75.233.244', now_time)
        backup_time = syn_backup(now_time)
        now_time = nextday_insert_startdata(now_time)   # 新一天开始来车
        update_data('106.75.233.244', now_time)
        now_request_trucks = cal_request_trucks()
        user_num = len(now_request_trucks)
        print(user_num)

    for truck in now_request_trucks:
        recordid = truck[0]
        user_code = truck[1]
        cargo_name = truck[2]
        start_time = truck[4]
        warehouse_name = truck[3]   # routing_list, now_time, start_time, warehouse_name
        predict_p, status1_p, status2_p, status3_p, exesql_num_p, _\
            = predict_wait_time([1, 1, 1, 1], now_time, start_time, warehouse_name, 1, 1)
        predict_s, status1_s, status2_s, status3_s, _, exesql_num_s \
            = predict_wait_time([0, 0, 0, 0], now_time, start_time, warehouse_name, 1, 1)

        # 保存执行的特征获取sql数
        with open('/root/dataset_accu/collect/expt_sql_num0.csv', 'a+') as sqlnumfile:
            result_sqlnum = [now_time, start_time, exesql_num_p, exesql_num_s]
            writer = csv.writer(sqlnumfile)
            writer.writerow(result_sqlnum)

        # 处理收集数据，按顺序放，补全
        sample_data = list()
        sample_data.extend(list(vgap_time.values()))  # 新鲜度特征, 分钟差值
        sample_data.extend(list(vgap_num.values()))
        sample_data.append(now_time)  # 辅助特征 （自带的，检查查询得到，动态计算得到）
        sample_data.append(recordid)
        sample_data.append(user_code)
        sample_data.append(start_time)
        # sample_data.append(cargo_name)   # 当前用户的加载时间不重要，主要是前面的车辆的加载时间
        sample_data.append(warehouse_name)
        # 以下是需要查询获取的，有些可能需要大量查询，辅助特征
        # 主备都查询
        park_count = list()
        for ip in ['106.75.233.244', '106.75.244.49']:
            db = pymysql.connect(host=ip, user='root', password='huangss123',
                                 database='accuqueuedata', charset='utf8')
            cursor = db.cursor()
            sql = "SELECT park_count FROM warehouse_info WHERE warehouse_name = %s;"
            cursor.execute(sql, (warehouse_name, ))
            park_count.append(cursor.fetchone()[0])  # 得到此库下可停放的车辆数
            cursor.close()
            db.close()
        sample_data.extend(park_count)  # 主、备的查询结果

        # 处理动态特征位数----用list保存. status3直接是一个值。前两个是list，其顺序就是排队的顺序
        sample_data.append(status1_p)
        sample_data.append(status1_s)
        all_status = [status2_p, status2_s]
        for s in all_status:  # 取十位，不足补零
            if len(s) < 10:
                processed_list = s[:10] + [0] * (10 - len(s))
            else:
                processed_list = s[:10]
            sample_data.append(processed_list)  # 注意，整个收集的数据中，就这个是个是list，且数据有10个
        sample_data.append(status3_p)
        sample_data.append(status3_s)

        # 预测结果值
        sample_data.append(predict_p)
        sample_data.append(predict_s)
        sample_data.append(abs(predict_p-predict_s))

        # 最后加个现在主备的时间差异，虽然版本表能看到，但是单独记比较方便观察
        gapt = (now_time-backup_time).total_seconds()
        sample_data.append(gapt/60)
        sample_data.append(gapt)

        # print(sample_data)
        writer = csv.writer(mydatafile, dialect='excel')
        writer.writerow(sample_data)

    user_request_all.append(user_num)  # 记录次轮gap收集到的数据条数
    if user_num == 0:
        print(now_time)

    # 收集到的数据，对应的列名及访问下标：
    # vtime_park_count, vtime_loading_time, vtime_start_time, vtime_entry_time, vtime_entry_whouse_time, vtime_finish_time,
    # vtime_queuing_info, vtime_cargo_info, vtime_warehouse_info, vtime_accuqueuedata   ---10   0-9
    # 同上前缀为 vnum_   ---10  10-19
    # now_time  ---1  20
    # recordid, user_code, start_time, warehouse_name  ---4  21-24
    # park_count_p, park_count_s    ---2  25-26
    # status1_p, status1_s, status2_p, status2_s  ---4  27-30  list
    # status3_p, status3_s  ---2  31-32
    # predict_p, predict_s  ---2  33-34
    # 主库时间差两个，秒和分钟的


def get_version():
    print("loading_time, park_count")
    for ip in ['106.75.233.244', '106.75.244.49']:
        db = pymysql.connect(host=ip, user='root', password='huangss123',
                             database='accuqueueversion', charset='utf8')
        cursor = db.cursor()
        version = list()
        sql = "SELECT * from vnum_loading_time;"
        cursor.execute(sql)
        version.append(cursor.fetchone()[0])
        sql = "SELECT * from vtime_loading_time;"
        cursor.execute(sql)
        version.append(cursor.fetchone()[0])
        sql = "SELECT * from vnum_park_count;"
        cursor.execute(sql)
        version.append(cursor.fetchone()[0])
        sql = "SELECT * from vtime_park_count;"
        cursor.execute(sql)
        version.append(cursor.fetchone()[0])
        cursor.execute(sql)
        db.commit()
        print(f"{version}")
        cursor.close()
        db.close()

def syn_backup(backup_time):
    queue_wk.queuing_syn()  # 包含对库下的三个表的同步，使用SQL语句同步。新补充tavlea的同步
    version_syn()
    return backup_time

# 为了测试，查看每次gap后可以发送请求的用户的id （是recordid，不是user_code，方便看）
def get_request_id():
    now_request_trucks = cal_request_trucks()
    print(f"现在可发请求用户数：{len(now_request_trucks)}")
    Urecordid = list()
    for uinfo in now_request_trucks:
        Urecordid.append(uinfo[0])
    print(f"{Urecordid}")

# 临时观看样本数据，用来调整负载和检测收集质量
def view_sample():
    mydata = pd.read_csv('/root/dataset_accu/collect/sample_data.csv', header=None, names=sample_data_colums)
    fig = plt.figure(figsize=(8, 4))
    # mydata = mydata[mydata['error'] < 25000]
    # mydata = mydata[mydata['vgap'] > 1 ]
    # mydata = mydata[mydata['error'] < 200]

    x = mydata['gap_time']
    y = mydata['error']

    # color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    plt.scatter(x, y)

    plt.xlabel("vgap")
    plt.ylabel("error")

    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig("/root/dataset_accu/collect/figsample.pdf", format="pdf")

if __name__ == "__main__":
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
    warehousedata_to_mysql()
    cargodata_to_mysql()
    create_versiontable()
    gapnum = 1500  # 按比例取。改动需要注意QueueWorkload2中一些相关数据file设置的值。create_staticfile函数里
    user_request_all = list()   # 记录每个gap时可以发送请求的数量，记载文件里
    queue_wk = QueueWorkload2(1, gapnum)  # 一个空queuing_info表，空的无关表tablea

    # 负载开始 (这个时间不随意修改，version初始化时会使用，初始车辆也是这个时间为base)
    # primary_time = datetime.strptime('2021-01-01 05:59:50', '%Y-%m-%d %H:%M:%S')
    # backup_time = datetime.strptime('2021-01-01 05:59:50', '%Y-%m-%d %H:%M:%S')
    mydatafile = open('/root/dataset_accu/collect/sample_data.csv', 'w+')
    # writer = csv.writer(mydatafile, dialect='excel')  # 加列名
    # writer.writerow(sample_data_colums)

    # 基础数据 （主备都有这些数据）
    now_time = datetime.strptime('2021-01-01 05:59:50', '%Y-%m-%d %H:%M:%S')
    queue_wk.create_queuedata([1], 5, now_time.strftime('%Y-%m-%d %H:%M:%S'), 10)  # 仓库编号，车辆数，到达时间base，时间间隔
    queue_wk.create_queuedata([2], 3, '', 20)  # 时间参数为空，会获取last记录的时间的start_time作为base
    queue_wk.create_queuedata([1], 4, '', 5)
    queue_wk.create_queuedata([5], 6, '', 8)
    queue_wk.create_queuedata([2], 4, '', 5)
    queue_wk.create_queuedata([5], 2, '', 5)
    queue_wk.create_queuedata([1], 2, '', 5)
    add_time = 5 * 10 + 3 * 20 + 4 * 5 + 6 * 8 + 4 * 5 + 2 * 5 + 2 * 5
    now_time += timedelta(seconds=add_time)
    update_data('106.75.233.244', now_time)  # 更新主库状态
    backup_time = syn_backup(now_time)  # 同步备库
    # print(f"开始操作：{now_time}")


    for i in range(1, gapnum+1):
        print(f"*****{i}*****")
        now_time = queue_wk.dynmaicload(now_time, i)  # 插入新车
        update_data('106.75.233.244', now_time)  # 更新主库
        collect_sample()  # 收集数据
        backup_time = syn_backup(now_time)  # 同步备库，进入下轮收集

    mydatafile.close()

    queue_wk.view_classdata()
    print(f"现在最后的停止时间：{now_time}")

    # 每次gap可请求数记录
    with open("/root/dataset_accu/collect/user_request_times.txt", 'w+') as file:
        file.write(str(user_request_all))
    # with open("/root/dataset_accu/collect/user_request_times.txt", 'r') as file:
    #     user_request_all = ast.literal_eval(file.readline().rstrip())


