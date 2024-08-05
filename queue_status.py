import ast
import csv

import pymysql
from datetime import datetime, timedelta

### 获取状态，等待时间预测、数据更新


# 这个函数只获取排在“我”队伍前、库外等候、车库工作，这些车辆的还需工作时间
# 改动+++
# 因为在感知服务预测时，收集各种预测结果，所以这里添加路由list，目前指定4为值，因为预测只需要这四位来计算。注意第一个值在predict_wait_time函数
# 计算每个请求全部到主、被、感知路由的真实计算的结果
# 添加在于不同在于添加了一个指定路由，以决定单个查询去主还是去备
# 最后一个变量，用来表示是否记录sal语句的，因为会调用很多次，但是实际只需要记录一下
def cal_status(routing_list, now_time, warehouse_name, start_time, flag_save_sql=0, flag_exesql_num=0):
    db_p = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor_p = db_p.cursor()
    db_s = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor_s = db_s.cursor()

    # 需要记录执行次数
    exe1p = 0
    exe2p = 0
    exe3p = 0
    exe1s = 0
    exe2s = 0
    exe3s = 0

    # 1.队伍前：同仓,start在我的start之前，entry为空
    # 另外需注意，由于更新延迟，entry大于now也表示正处于队伍前
    if routing_list[1] == 1:
        cursor = cursor_p
    else:
        cursor = cursor_s
    sql = "SELECT cargo_name, recordid FROM queuing_info WHERE warehouse_name = %s " \
          "AND start_time < %s AND " \
          "(entry_time IS NULL OR entry_time > %s);"  # 加一个recordid后期没用，只是一开始测试使用
    cursor.execute(sql, (warehouse_name, start_time, now_time))

    # 保存sql
    if flag_save_sql == 1:
        getsql = cursor.mogrify(sql, (warehouse_name, start_time, now_time))
        if routing_list[1] == 1:
            with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                writer = csv.writer(sqlpfile)
                writer.writerow([now_time, start_time, getsql])
        else:
            with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                writer = csv.writer(sqlsfile)
                writer.writerow([now_time, start_time, getsql])


    r1 = cursor.fetchall()
    #第一个条件表示要记录此时运行的sql数
    if flag_exesql_num == 1 and routing_list[1] == 1:
        exe1p += (len(r1) + 1)
    elif flag_exesql_num == 1 and routing_list[1] == 0:
        exe1s += (len(r1) + 1)

    truck1 = list()  # 符合要求的车的标识
    loadingtime_list1 = list()  #他们的loading时间，后续算等待时间会用到
    # 如果没有结果，前面没车。否则计算前面车数及他们所需工作时间
    if r1:
        for t in r1:
            sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
            cursor.execute(sql, (t[0], ))

            # 保存sql
            if flag_save_sql == 1:
                getsql = cursor.mogrify(sql, (t[0], ))
                if routing_list[1] == 1:
                    with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                        writer = csv.writer(sqlpfile)
                        writer.writerow([now_time, start_time, getsql])
                else:
                    with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                        writer = csv.writer(sqlsfile)
                        writer.writerow([now_time, start_time, getsql])

            loadingtime_list1.append(cursor.fetchone()[0])  # 正好这个查询出来的顺序就是前后排队的前后顺序
            truck1.append(t[1])
    # print("***第一阶段：队伍前")
    # print(f"车辆标识：{truck1}，需加载时间：{loadingtime_list1}")
    # 2.库外停车：entry在我now之前，entry_whous为空
    # （需要注意，这个查出的数可能大于可停车数，这是因为后期这个特征变化了，此时按照原来的基础算）
    # 条件，start小于我的start，entry要小于等于now，表示时间变化后入厂的车（注意，此处若是entry小于我的start，那就是原始状态，没有随时间变化）
    # 另外，entry_wh大于now，也表示还在排队中
    if routing_list[2] == 1:
        cursor = cursor_p
    else:
        cursor = cursor_s
    sql = "SELECT cargo_name, recordid FROM queuing_info WHERE warehouse_name = %s " \
          "AND start_time < %s AND entry_time IS NOT NULL AND entry_time <= %s AND" \
          "(entry_whouse_time IS NULL OR entry_whouse_time > %s);"
    cursor.execute(sql, (warehouse_name, start_time, now_time, now_time))

    # 保存sql
    if flag_save_sql == 1:
        getsql = cursor.mogrify(sql, (warehouse_name, start_time, now_time, now_time))
        if routing_list[1] == 1:
            with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                writer = csv.writer(sqlpfile)
                writer.writerow([now_time, start_time, getsql])
        else:
            with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                writer = csv.writer(sqlsfile)
                writer.writerow([now_time, start_time, getsql])

    r2 = cursor.fetchall()
    if flag_exesql_num == 1 and routing_list[2] == 1:
        exe2p += (len(r2) + 1)
    elif flag_exesql_num == 1 and routing_list[2] == 0:
        exe2s += (len(r2) + 1)

    truck2 = list()
    loadingtime_list2 = list()
    if r2:
        for t in r2:
            sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
            cursor.execute(sql, (t[0], ))

            # 保存sql
            if flag_save_sql == 1:
                getsql = cursor.mogrify(sql, (t[0], ))
                if routing_list[1] == 1:
                    with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                        writer = csv.writer(sqlpfile)
                        writer.writerow([now_time, start_time, getsql])
                else:
                    with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                        writer = csv.writer(sqlsfile)
                        writer.writerow([now_time, start_time, getsql])

            loadingtime_list2.append(cursor.fetchone()[0])
            truck2.append(t[1])
    # print("***第二阶段：库外停车数")
    # print(f"车辆标识：{truck2}，需加载时间：{loadingtime_list2}")
    # 3.库内工作: whouse在我now之前，finish为空
    # (这个车的已工作时间，还需工作时间）
    # entry_wh小于等于now，随时间变换后已经入库的车。 finish为NULL，表示还在工作。
    # 另外，finish大于now也是表示在工作。这是因此此时是延迟的数据更新，所以才会finish不为空
    if routing_list[3] == 1:
        cursor = cursor_p
    else:
        cursor = cursor_s
    sql = "SELECT cargo_name, entry_whouse_time, recordid FROM queuing_info WHERE warehouse_name = %s " \
          "AND start_time < %s AND entry_whouse_time IS NOT NULL AND entry_whouse_time <= %s AND" \
          "(finish_time IS NULL OR finish_time > %s);"
    cursor.execute(sql, (warehouse_name, start_time, now_time, now_time))

    # 保存sql
    if flag_save_sql == 1:
        getsql = cursor.mogrify(sql, (warehouse_name, start_time, now_time, now_time))
        if routing_list[1] == 1:
            with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                writer = csv.writer(sqlpfile)
                writer.writerow([now_time, start_time, getsql])
        else:
            with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                writer = csv.writer(sqlsfile)
                writer.writerow([now_time, start_time, getsql])

    r3 = cursor.fetchone()
    if flag_exesql_num == 1 and routing_list[3] == 1:
        if r3:
            exe3p += 2
    elif flag_exesql_num == 1 and routing_list[3] == 0:
        if r3:
            exe3s += 2

    truck3 = 0
    remain_loadingtime = 0
    if r3:
        truck3 = r3[2]
        sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
        cursor.execute(sql, (r3[0], ))

        # 保存sql
        if flag_save_sql == 1:
            getsql = cursor.mogrify(sql, (r3[0], ))
            if routing_list[1] == 1:
                with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                    writer = csv.writer(sqlpfile)
                    writer.writerow([now_time, start_time, getsql])
            else:
                with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                    writer = csv.writer(sqlsfile)
                    writer.writerow([now_time, start_time, getsql])

        need_loadtime = cursor.fetchone()[0]
        worked_time = (now_time - r3[1]).total_seconds() / 60   # 计算已经工作时间
        remain_loadingtime = need_loadtime - worked_time
        if remain_loadingtime < 0:
            remain_loadingtime = 0

    # print("***第三阶段：库内工作")
    # print(f"车辆标识：{truck3}，剩余需工作时间：{remain_loadingtime}")

    cursor_p.close()
    db_p.close()
    cursor_s.close()
    db_s.close()

    # 预测请求中，四个特征可路由
    # 第一个是predict_wait_time函数里的，一条sql
    # 其余三个，包含固定一条外，剩余看对应前面车辆个数。这个函数中统计了
    exesql_num_p = [1, exe1p, exe2p, exe3p]
    exesql_num_s = [1, exe1s, exe2s, exe3s]

    # remain用来预测等待时间。lastfinish用来数据更新时使用。
    # 这些list的顺序就是真实队伍的顺序
    return loadingtime_list1, loadingtime_list2, remain_loadingtime, exesql_num_p, exesql_num_s



# 暂时不用，同下一个函数的2，增加了返回值
def cal_status2(ip, now_time, warehouse_name, start_time):
    db = pymysql.connect(host=ip, user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()

    # 1.队伍前：同仓,start在我的start之前，entry为空
    # 另外需注意，由于更新延迟，entry大于now也表示正处于队伍前
    sql = "SELECT cargo_name, recordid FROM queuing_info WHERE warehouse_name = %s " \
          "AND start_time < %s AND " \
          "(entry_time IS NULL OR entry_time > %s);"  # 加一个recordid后期没用，只是一开始测试使用
    cursor.execute(sql, (warehouse_name, start_time, now_time))
    r1 = cursor.fetchall()
    truck1 = list()  # 符合要求的车的标识
    loadingtime_list1 = list()  #他们的loading时间，后续算等待时间会用到
    cargo_name_list1 = list()   # 前状态车辆的货物名也记录下来，用在负载区分的时候
    # 如果没有结果，前面没车。否则计算前面车数及他们所需工作时间
    if r1:
        for t in r1:
            sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
            cursor.execute(sql, (t[0], ))
            loadingtime_list1.append(cursor.fetchone()[0])  # 正好这个查询出来的顺序就是前后排队的前后顺序
            truck1.append(t[1])
            cargo_name_list1.append(t[0])
    # print("***第一阶段：队伍前")
    # print(f"车辆标识：{truck1}，需加载时间：{loadingtime_list1}")
    # 2.库外停车：entry在我now之前，entry_whous为空
    # （需要注意，这个查出的数可能大于可停车数，这是因为后期这个特征变化了，此时按照原来的基础算）
    # 条件，start小于我的start，entry要小于等于now，表示时间变化后入厂的车（注意，此处若是entry小于我的start，那就是原始状态，没有随时间变化）
    # 另外，entry_wh大于now，也表示还在排队中
    sql = "SELECT cargo_name, recordid FROM queuing_info WHERE warehouse_name = %s " \
          "AND start_time < %s AND entry_time IS NOT NULL AND entry_time <= %s AND" \
          "(entry_whouse_time IS NULL OR entry_whouse_time > %s);"
    cursor.execute(sql, (warehouse_name, start_time, now_time, now_time))
    r2 = cursor.fetchall()
    truck2 = list()
    loadingtime_list2 = list()
    cargo_name_list2 = list()
    if r2:
        for t in r2:
            sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
            cursor.execute(sql, (t[0], ))
            loadingtime_list2.append(cursor.fetchone()[0])
            truck2.append(t[1])
            cargo_name_list2.append(t[0])
    # print("***第二阶段：库外停车数")
    # print(f"车辆标识：{truck2}，需加载时间：{loadingtime_list2}")
    # 3.库内工作: whouse在我now之前，finish为空
    # (这个车的已工作时间，还需工作时间）
    # entry_wh小于等于now，随时间变换后已经入库的车。 finish为NULL，表示还在工作。
    # 另外，finish大于now也是表示在工作。这是因此此时是延迟的数据更新，所以才会finish不为空
    sql = "SELECT cargo_name, entry_whouse_time, recordid FROM queuing_info WHERE warehouse_name = %s " \
          "AND start_time < %s AND entry_whouse_time IS NOT NULL AND entry_whouse_time <= %s AND" \
          "(finish_time IS NULL OR finish_time > %s);"
    cursor.execute(sql, (warehouse_name, start_time, now_time, now_time))
    r3 = cursor.fetchone()
    truck3 = 0
    remain_loadingtime = 0
    cargo_name_3 = ''
    if r3:
        truck3 = r3[2]
        sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
        cursor.execute(sql, (r3[0], ))
        need_loadtime = cursor.fetchone()[0]
        worked_time = (now_time - r3[1]).total_seconds() / 60   # 计算已经工作时间
        remain_loadingtime = need_loadtime - worked_time
        cargo_name_3 = r3[0]

    # print("***第三阶段：库内工作")
    # print(f"车辆标识：{truck3}，剩余需工作时间：{remain_loadingtime}")

    cursor.close()
    db.close()

    # remain用来预测等待时间。lastfinish用来数据更新时使用。
    # 这些list的顺序就是真实队伍的顺序
    return loadingtime_list1, loadingtime_list2, remain_loadingtime, \
           cargo_name_list1, cargo_name_list2, cargo_name_3

# 计算等待时间，首先调用 def cal_status()获取当前的队伍情况
# 这里的两个返回值是库外状态和库内剩余时间，为了在更新数据函数中方便使用，仅为了预测
# 改动+++
# 同使用的cal_status函数，为了指定路由方向，这里是list的第一项，park_count特征的
# 最后一个变量，用来表示是否记录sal语句的，因为会调用很多次，但是实际只需要记录一下
def predict_wait_time(routing_list, now_time, start_time, warehouse_name, flag_save_sql=0, flag_exesql_num=0):
    # 获取队列情况
    # print(start_time, type(start_time))
    loadingtime_list1, loadingtime_list2, remain_loadingtime, exesql_num_p, exesql_num_s \
        = cal_status(routing_list, now_time, warehouse_name, start_time, flag_save_sql, flag_exesql_num)

    if routing_list[0] == 1:
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                               database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
    else:
        db = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                               database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
    # 计算入厂等待时间
    sql = "SELECT park_count FROM warehouse_info WHERE warehouse_name = %s;"
    cursor.execute(sql, (warehouse_name, ))

    # 保存sql
    if flag_save_sql == 1:
        getsql = cursor.mogrify(sql, (warehouse_name, ))
        if routing_list[0] == 1:
            with open('/root/dataset_accu/collect/expt_sql_p.csv', 'a+') as sqlpfile:
                writer = csv.writer(sqlpfile)
                writer.writerow([now_time, start_time, getsql])
        else:
            with open('/root/dataset_accu/collect/expt_sql_s.csv', 'a+') as sqlsfile:
                writer = csv.writer(sqlsfile)
                writer.writerow([now_time, start_time, getsql])

    park_count = cursor.fetchone()[0]   #得到此库下可停放的车辆数
    cursor.close()
    db.close()

    front_car_num = len(loadingtime_list1) + len(loadingtime_list2)  # 排队我之前的车的数量，不包含库内的
    if remain_loadingtime == 0:
        front_car_num -= 1
    # 如果此时的在我之前的车数是小于停车数的，说明不用等待，返回时间需等待时间为0
    # （这个原因是停车数更新为更大的）
    if park_count > front_car_num:
        wait_time = 0
    elif park_count == front_car_num:  # 正好停满的情况。看多久库外能空出一位来
        wait_time = remain_loadingtime  # 库内车完成就可以空出一位
    else:  # 这应该是停车位属性被改小了，或者是队伍前有车。按停车位最前位开始加时间，数量是这两个的差
        wait_time = remain_loadingtime
        front_cat_list = loadingtime_list2 + loadingtime_list1
        wait_car_num = front_car_num - park_count
        # print(f"最后结果：需要等候{wait_car_num}车+1（库里）才轮到")
        for i in range(wait_car_num):
            wait_time += front_cat_list[i]
    # print(f"计算的等待时间: {wait_time} 分钟")
    # print("###########################")

    # 这里调整一下loadingtime_list1。但实际上这个厂外排队车辆数是不确定的。
    # 所以这个值仍然是list，但只包含两位，  厂外前车等待数和等待时间求和
    front_queuing_num = len(loadingtime_list1)
    front_queuing_loading = sum(loadingtime_list1)
    loadingtime_list1_new = [front_queuing_num, front_queuing_loading]

    return wait_time, loadingtime_list1_new, loadingtime_list2, remain_loadingtime, exesql_num_p, exesql_num_s

# 这个暂时不用，与上面的区别在于返回值个数不一样，这个加了三个返回前状态的车的货物
# 是为了区分预测查询和负载改动的，但是后期没用到，先不删除
def predict_wait_time2(ip, now_time, start_time, warehouse_name):
    db = pymysql.connect(host=ip, user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()

    # 获取队列情况
    # print(start_time, type(start_time))
    loadingtime_list1, loadingtime_list2, remain_loadingtime, \
    cargo_name_list1, cargo_name_list2, cargo_name_3 \
        = cal_status(ip, now_time, warehouse_name, start_time)

    # 计算入厂等待时间
    sql = "SELECT park_count FROM warehouse_info WHERE warehouse_name = %s;"
    cursor.execute(sql, (warehouse_name, ))
    park_count = cursor.fetchone()[0]   #得到此库下可停放的车辆数

    front_car_num = len(loadingtime_list1) + len(loadingtime_list2)  # 排队我之前的车的数量，不包含库内的
    # 如果此时的在我之前的车数是小于停车数的，说明不用等待，返回时间需等待时间为0
    # （这个原因是停车数更新为更大的）
    if park_count > front_car_num:
        wait_time = 0
    elif park_count == front_car_num:  # 正好停满的情况。看多久库外能空出一位来
        wait_time = remain_loadingtime  # 库内车完成就可以空出一位
    else:  # 这应该是停车位属性被改小了，或者是队伍前有车。按停车位最前位开始加时间，数量是这两个的差
        wait_time = remain_loadingtime
        front_cat_list = loadingtime_list2 + loadingtime_list1
        wait_car_num = front_car_num - park_count
        # print(f"最后结果：需要等候{wait_car_num}车+1（库里）才轮到")
        for i in range(wait_car_num):
            wait_time += front_cat_list[i]
    # print(f"计算的等待时间: {wait_time} 分钟")
    # print("###########################")

    cursor.close()
    db.close()

    return wait_time, loadingtime_list1, loadingtime_list2, remain_loadingtime, \
           cargo_name_list1, cargo_name_list2, cargo_name_3

#根据时间更新排队表.除了自行指定时间，还有根据每个的start_time可更新
# 更新方法，start在更新时间前，且其中一个时间属性为null，表示需要更新，进行：
# 1）判断entry_time 为 空的
#   计算各个时间属性，在更新时间内则更新
# 2）判断entry_wh 为空
# 3）判断finish 为 空的
def update_data(ip, update_time):
    db = pymysql.connect(host=ip, user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()
    # 这里对start使用小于等于，是因为场景设置中省略掉了开车的过程，所有有些属性时间的值是可以一样的
    sql = "SELECT start_time, cargo_name, entry_time, entry_whouse_time, user_code, warehouse_name FROM queuing_info " \
          "WHERE start_time <= %s AND " \
          "(entry_time IS NULL OR entry_whouse_time IS NULL OR finish_time IS NULL);"
    cursor.execute(sql, (update_time, ))
    result = cursor.fetchall()
    # print("进入判断的条目")
    # for i in result:
    #     print(i)
    # insert_usercode = list()
    if result:  # 没有结果则不用更新
        for row in result:
            # start_time = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            start_time = row[0]
            cargo_name = row[1]
            entry_time = row[2]
            entry_whouse_time = row[3]
            user_code = row[4]
            warehouse_name = row[5]
            # 将start作为nowtime去计算当前的状态和需等待时间
            wait_time, loadingtime_list1, loadingtime_list2, remain_loadingtime, _, _ = \
                predict_wait_time([1, 1, 1, 1], start_time, start_time, warehouse_name)

            if not entry_time :   # 1)
                # 应该的进入时间是否小于当前时间
                new_entry_time = start_time + timedelta(minutes=wait_time)
                if new_entry_time <= update_time:  #如果不满足，则后面的其他属性时间都不用计算了
                    sql = "UPDATE queuing_info SET entry_time = %s WHERE user_code = %s;"
                    params = (new_entry_time, user_code)
                    filled_sql = cursor.mogrify(sql, params)
                    # print(filled_sql)
                    cursor.execute(sql, params)
                    db.commit()
                    # insert_usercode.append(user_code)
                    # 计算entry_wh的时间
                    # 注意这里只需要算停车的总时间，因为库内车的时间药已经包含在我的entry中
                    sum_loading_time = sum(loadingtime_list1) + sum(loadingtime_list2) + remain_loadingtime
                    new_entry_whouse_time = start_time + timedelta(minutes=sum_loading_time)
                    if new_entry_whouse_time <= update_time:
                        sql = "UPDATE queuing_info SET entry_whouse_time = %s WHERE user_code = %s;"
                        params = (new_entry_whouse_time, user_code)
                        filled_sql = cursor.mogrify(sql, params)
                        # print(filled_sql)
                        cursor.execute(sql, params)
                        db.commit()
                        # insert_usercode.append(user_code)
                        # 计算finish时间
                        # 将要更新的finish_time还需要加上“我”的加载时间
                        sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
                        cursor.execute(sql, (cargo_name,))
                        my_loadtime = cursor.fetchone()[0]
                        new_finish_time = new_entry_whouse_time + timedelta(minutes=my_loadtime)
                        if new_finish_time <= update_time:
                            sql = "UPDATE queuing_info SET finish_time = %s WHERE user_code = %s;"
                            params = (new_finish_time, user_code)
                            filled_sql = cursor.mogrify(sql, params)
                            # print(filled_sql)
                            cursor.execute(sql, params)
                            db.commit()
                            # insert_usercode.append(user_code)
            elif not entry_whouse_time: # 2)
                # 注意只算finish，可以根据entery_wh来算。但是算entry_wh，由于停车位不同，减少判断，所以根据当前状态用start比较方便
                sum_loading_time = sum(loadingtime_list1) + sum(loadingtime_list2) + remain_loadingtime
                new_entry_whouse_time = start_time + timedelta(minutes=sum_loading_time)
                if new_entry_whouse_time <= update_time:
                    sql = "UPDATE queuing_info SET entry_whouse_time = %s WHERE user_code = %s;"
                    params = (new_entry_whouse_time, user_code)
                    filled_sql = cursor.mogrify(sql, params)
                    # print(filled_sql)
                    cursor.execute(sql, params)
                    db.commit()
                    # insert_usercode.append(user_code)
                    # 计算finish时间
                    sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
                    cursor.execute(sql, (cargo_name,))
                    my_loadtime = cursor.fetchone()[0]
                    new_finish_time = new_entry_whouse_time + timedelta(minutes=my_loadtime)
                    if new_finish_time <= update_time:
                        sql = "UPDATE queuing_info SET finish_time = %s WHERE user_code = %s;"
                        params = (new_finish_time, user_code)
                        filled_sql = cursor.mogrify(sql, params)
                        # print(filled_sql)
                        cursor.execute(sql, params)
                        db.commit()
                        # insert_usercode.append(user_code)
            else: # 3)
                sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
                cursor.execute(sql, (cargo_name,))
                my_loadtime = cursor.fetchone()[0]
                new_finish_time = entry_whouse_time + timedelta(minutes=my_loadtime)
                if new_finish_time <= update_time:
                    sql = "UPDATE queuing_info SET finish_time = %s WHERE user_code = %s;"
                    params = (new_finish_time, user_code)
                    filled_sql = cursor.mogrify(sql, params)
                    # print(filled_sql)
                    cursor.execute(sql, params)
                    db.commit()
                    # insert_usercode.append(user_code)

    # 将记录的修改时保存的usercode，获取对应的仓库名。用作负载判断时使用
    # insert_usercode_set = list(set(insert_usercode))
    # insert_warehouse = list()
    # for u in insert_usercode_set:
    #     sql = "SELECT warehouse_name FROM queuing_info WHERE user_code=%s;"
    #     cursor.execute(sql, (u, ))
    #     r = cursor.fetchone()[0]
    #     insert_warehouse.append(r)

    cursor.close()
    db.close()
    # return insert_warehouse


# 目前可发出预测请求的车辆
# 以主库数据为准，在厂外排队的都可以
def cal_request_trucks():
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()

    sql = "SELECT recordid, user_code, cargo_name, warehouse_name, start_time FROM queuing_info WHERE entry_time IS NULL;"
    # sql = "SELECT recordid, user_code, cargo_name, warehouse_name, start_time FROM queuing_info " \
    #       "WHERE (entry_time IS NULL OR entry_time > %s) AND start_time <= %s;"
    # cursor.execute(sql, (now_time, now_time))
    # 把warehouse id加到结果中，为了方便后面模型训练和路由使用，因为使用中文还需要编码（这里去掉了，后面还是用的nname转编码）
    # sql = "SELECT qi.recordid, qi.user_code, qi.cargo_name, qi.warehouse_name, qi.start_time, wi.warehouseid " \
    #       "FROM queuing_info qi JOIN warehouse_info wi ON qi.warehouse_name = wi.warehouse_name " \
    #       "WHERE qi.entry_time IS NULL;"
    cursor.execute(sql)
    result = cursor.fetchall()
    now_request_trucks = list()
    if result:
        for r in result:
            now_request_trucks.append(list(r))
    # print(now_request_trucks)
    return now_request_trucks



# 这里可以随便修改，测试一些小操作
def test_ttt():
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()
    sql = 'select * from warehouse_info where warehouseid = 1;'
    cursor.execute(sql)
    r = cursor.fetchone()
    cargo_list = ast.literal_eval(r[3])
    print(cargo_list)
    print(type(cargo_list))

    cursor.close()
    db.close()


if __name__ == '__main__':
    # test_ttt()

    ####### 状态获取 ######
    # 如果使用的时间不是直接数据库查出来，而是自己写的字符串的，则需要转换
    ip = '106.75.233.244'
    now_time = datetime.strptime('2021-01-01 06:05:40', '%Y-%m-%d %H:%M:%S')
    warehouse_name = "#1中间板(即热轧板)库"
    start_time = datetime.strptime('2021-01-01 06:01:55', '%Y-%m-%d %H:%M:%S')
    # 获取此时在”我“前面的车在各个阶段的状态
    loadingtime_list1, loadingtime_list2, remain_loadingtime = \
        cal_status(ip, now_time, warehouse_name, start_time)
    print(loadingtime_list1)
    print(loadingtime_list2)
    print(remain_loadingtime)
    # print(cargo_name_list1)
    # print(cargo_name_list2)
    # print(cargo_name_3)


    ####### 等候时间预测 ######
    # user_code = 'U000'
    # user_code用来查start_time和warehouse_name，如果不提前查询的话，备库可能没有这个用户的信息
    # sql = "SELECT warehouse_name, start_time FROM queuing_info WHERE user_code = %s;"
    # cursor.execute(sql, (user_code,))
    # r = cursor.fetchone()
    # warehouse_name = r[0]
    # start_time = r[1]
    # cal_time, _ = predict_wait_time(ip, now_time, start_time, warehouse_name)
    # print(cal_time)

    ####### 排队数据更新 ######
    # t = datetime.strptime('2021-01-01 05:07:15', '%Y-%m-%d %H:%M:%S')  # 直接传入的字符串时间记得需要进行格式转换
    # update_data(ip, t)

    # start_time = datetime.strptime('2021-01-01 05:00:20', '%Y-%m-%d %H:%M:%S')  # 直接传入的字符串时间记得需要进行格式转换
    # loadingtime_list1, loadingtime_list2, remain_loadingtime = \
    #         cal_status(ip, start_time, '小棒库(二棒)', start_time)

    # 可请求预测的车辆
    # now_time = datetime.strptime('2021-01-01 06:20:00', '%Y-%m-%d %H:%M:%S')
    # now_request_trucks = cal_request_trucks(now_time)
