import random
import threading
import time
import numpy as np
import pymysql
import pandas as pd
import ast
from datetime import datetime, timedelta




### 排队
# 生成一些基本数据，一段时间累积来车辆create_basedata()这个函数原本来自create_data.py中的create_queuing修改
# 运行过程中缓慢生成数据
# 更改 停车数、加载时间
# 这一版是data_collect4之前的使用的，因为这之前的都没有把插入车，修改静态变量的函数加入
from Wait.parameter import sql_create_queuing_info, sql_create_tablea, sql_create_warehouse_info, sql_create_cargo_info, \
    sql_insert_queuing_info, sql_insert_warehouse_info, sql_insert_cargo_info, sql_update_col, sql_insert_tablea


class QueueWorkload:
    def __init__(self):
        self.user_num = 1   #车辆名不会重复，一直生成 UXXX 三位数的形式
        self.create_queuing_table()

    #初始化负载对象操作一次  创建一个空表
    def create_queuing_table(self):
        for ip in ['106.75.233.244', '106.75.244.49']:
            db = pymysql.connect(host=ip, user='root', password='huangss123',
                                 database='accuqueuedata', charset='utf8')
            cursor = db.cursor()
            sql = "DROP TABLE IF EXISTS queuing_info;"
            cursor.execute(sql)
            db.commit()
            sql = "CREATE TABLE queuing_info (recordid int NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
                  "user_code varchar(10), cargo_name varchar(255), warehouse_name varchar(255), " \
                  "start_time datetime, entry_time datetime, entry_whouse_time datetime, finish_time datetime);"
            cursor.execute(sql)
            db.commit()
            cursor.close()
            db.commit()

    # 获取上一条记录的开始时间
    def get_last_start_time(self):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        sql = "SELECT start_time FROM queuing_info ORDER BY recordid DESC LIMIT 1;"
        cursor.execute(sql)
        last_start_time = cursor.fetchone()[0]
        cursor.close()
        db.commit()

        return last_start_time

    # 生成累积
    # 不考虑前序排位信息，只是根据指定生成指定仓库的排队排队车，车的货物是按仓库kind循环的
    # 参数： 仓库编号，仓库下车车辆数，到达时间（如果为空就接上一条记录的时间，每辆到达的间隔
    def create_queuedata(self, warehouse_num, num, come_time, sec):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()

        # 初始化，userid是 U000，按self.user_num中起始的顺序
        # 根据指定仓库和排队数量，物品循环，然后打乱放进去再加算时间
        queue_df = pd.DataFrame(columns=['user_code', 'cargo_name', 'warehouse_name'])
        # warehouse_num = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        # warehouse_num = [6, 21]
        # num = 15
        cargo_all = list()  # 总的初始化，包含指定的所有的仓库
        warehouse_all = list()
        for wid in warehouse_num:
            sql = "select warehouse_name, cargo_kind from warehouse_info where warehouseid = %s;"
            cursor.execute(sql, (wid,))
            r = cursor.fetchone()
            kind = ast.literal_eval(r[1])
            # 根据当前仓库和货物列表生成的list。货物循环放置
            ware_list = [r[0] for _ in range(num)]
            kind_list = list()
            for i in range(num):
                kind_list.append(kind[i % len(kind)])
            # print(ware_list)
            # print(kind_list)

            cargo_all.extend(kind_list)
            warehouse_all.extend(ware_list)

        # 补充usercode
        user_all = list()
        start = self.user_num
        end = self.user_num + len(warehouse_num) * num
        self.user_num = end
        for i in range(start, end):
            user_all.append(f"U{i:06}")

        # 初始化数据组合并打乱
        queue_df['user_code'] = user_all
        queue_df['cargo_name'] = cargo_all
        queue_df['warehouse_name'] = warehouse_all
        # print(queue_df)
        # 打乱
        shuffled_df = queue_df.sample(frac=1, random_state=33)
        # print(shuffled_df)
        # 重现
        # original_order_df = queue_df.sample(frac=1, random_state=33)
        # print(original_order_df)

        # 直接将开始时间补充到数据库（因为数据是按时间顺序到的）
        start_time_all = list()
        if come_time:   # 不为空
            current_time = datetime.strptime(come_time, '%Y-%m-%d %H:%M:%S')  # 解析起始时间字符串
        else:
            current_time = self.get_last_start_time()
        for _ in range(len(user_all)):
            current_time += timedelta(seconds=sec)  # 增加指定的时间间隔
            start_time_all.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))  # 格式化当前时间并添加到列表中
        shuffled_df['start_time'] = start_time_all
        # print(shuffled_df)

        # 插入数据
        # insert_warehouse = list()
        for _, r in shuffled_df.iterrows():
            sql = "INSERT INTO queuing_info (user_code, cargo_name, warehouse_name, start_time) VALUES(%s, %s, %s, %s);"
            cursor.execute(sql, (r['user_code'], r['cargo_name'], r['warehouse_name'], r['start_time']))
            # insert_warehouse.append(r['warehouse_name'])
        db.commit()

        ###### 生成少量数据，为了测试其他模块功能，后期可注释 ######
        # 这里格外添加保存在excel，是为了生成小一点的数据，手动进行修改再上传，测试实现的功能是否正确
        # 这一段后期不用了，可以注释掉
        # sql = "SELECT * FROM queuing_info;"
        # queuing_df = pd.read_sql_query(sql, db)  # 这个函数直接将查询结果生成dataframe
        # queuing_df.to_excel('/root/dataset_accu/queuing_info.xlsx', index=False, encoding='utf-8')
        #############

        cursor.close()
        db.close()

        # return insert_warehouse  # 用于负载的时候判断添加数据的仓库名。已经去重过。  注意：其实在自己插入数据的时候就指定了


    # 主库同步到备库，这里直接查询然后插入过去即可，包括库下的三个表
    def queuing_syn(self):
        primaryDB = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                    database='accuqueuedata', charset='utf8')
        secondaryDB = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                      database='accuqueuedata', charset='utf8')
        cursor_P = primaryDB.cursor()
        cursor_S = secondaryDB.cursor()

        # 1. 备库表均删除并重建
        sql = "DROP TABLE IF EXISTS queuing_info;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        sql = "CREATE TABLE queuing_info (recordid int NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
              "user_code varchar(10), cargo_name varchar(255), warehouse_name varchar(255), " \
              "start_time datetime, entry_time datetime, entry_whouse_time datetime, finish_time datetime);"
        cursor_S.execute(sql)
        secondaryDB.commit()
        sql = "DROP TABLE IF EXISTS warehouse_info;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        sql = "CREATE TABLE warehouse_info (warehouseid tinyint PRIMARY KEY, warehouse_name varchar(255), " \
              "park_count tinyint, cargo_kind varchar(255));"
        cursor_S.execute(sql)
        secondaryDB.commit()
        sql = "DROP TABLE IF EXISTS cargo_info;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        sql = "CREATE TABLE cargo_info (cargoid tinyint PRIMARY KEY, cargo_name varchar(255), " \
              "loading_time tinyint);"
        cursor_S.execute(sql)
        secondaryDB.commit()

        # 2. 主库的表sql查询全表后插入到备库
        sql = "SELECT * FROM queuing_info;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            sql = "INSERT INTO queuing_info VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
            values = (row['recordid'], row['user_code'], row['cargo_name'], row['warehouse_name'],
                      row['start_time'], row['entry_time'], row['entry_whouse_time'], row['finish_time'])
            # NaN不能直接当做参数值用到sql中，需要转为NULL
            new_vaules = tuple(None if x != x else x for x in values)
            cursor_S.execute(sql, new_vaules)
        sql = "SELECT * FROM warehouse_info;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            sql = "INSERT INTO warehouse_info VALUES (%s, %s, %s, %s);"
            values = (row['warehouseid'], row['warehouse_name'], row['park_count'], row['cargo_kind'])
            cursor_S.execute(sql, values)
        sql = "SELECT * FROM cargo_info;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            sql = "INSERT INTO cargo_info VALUES (%s, %s, %s);"
            values = (row['cargoid'], row['cargo_name'], row['loading_time'])
            cursor_S.execute(sql, values)

        secondaryDB.commit()
        cursor_P.close()
        cursor_S.close()
        primaryDB.close()
        secondaryDB.close()

    # 更改静态特征
    # 注意，操作后还有添加版本信息这个操作
    # def change_park_count(self):


# 这一版是data_collect4使用的，
# 新增：把插入车，修改静态变量加入。动态生成负载。
# 增加无关表tablea，列有标识tid列，t1、t2、t3、t4随机数，t5时间。读取最新行的t5即为本表的Time新鲜度，读取表的总行数即为Num新鲜度。
# 参数1：将一天的时间划分为车辆到来速度的快慢。为1会生成新文件，再读入作备用。否则直接读取
# 参数2：总的gap次数，用来计算static修改次数和时机的
class QueueWorkload2:
    def __init__(self, basedata, gapnum):
        self.user_num = 1   #车辆名不会重复，一直生成 UXXX 三位数的形式

        self.all_cargo = ['全品种', '新产品-白卷', '新产品-卷板', '新产品-冷板', '新产品-窄带',
                          '老区-卷板', '老区-型钢', '老区-线材', '老区-开平板', '老区-螺纹',
                          '矿渣粉', '精品卷', '卷板', '开平板', '线材']
        self.all_warehouse = ['1中间板_即热轧板_库', '2中间板_即热轧板_库', '3中间板_即热轧板_库', '4中间板_即热轧板_库',
                              '大棒库_一棒', '小棒库_二棒', '1热轧卷成品库', '2热轧卷成品库', '3热轧卷成品库', '4热轧卷成品库',
                              '成品中间库', '运输处临港东库', '运输处临港西库', '岚北码头直取库', '伟冠临港库', '东铁临港库',
                              '中瑞临港库', '大棒中间库', '大H型钢成品库', '冷轧成品库_1号门', '冷轧成品库_2号门',
                              '冷轧成品库_6号门', '冷轧成品库_7号门', '冷轧成品库_8号门', '冷轧成品库_10号门', '冷轧成品库_9号门',
                              '剪切成品库_3号门', '剪切成品库_4号门', '剪切成品库_5号门', '平整卷成品库_12号门',
                              '平整卷成品库_13号门', '小H型钢成品库', '高线库_一线', '多头盘螺库', '热轧_2150成品_二库',
                              '热轧_2150成品_三库', '热轧_2150成品_四库', '精整1_成品库', '开平1_2_成品库', '开平3_成品库',
                              '精整2_成品库', '联储线材', '联储卷板', '小棒库_二棒_下线库', '大H型钢成品库_下线库',
                              '小H型钢成品库_下线库', '高线库_下线库', '高线库_二线', '高线库_三线', '平整卷成品库_11号门']

        self.basedata = basedata
        self.gapnum = gapnum
        self.freq_fast = list()  # 每条数据更新时间
        self.freq_slow = list()
        self.freq_veryslow = list()
        self.fast_i = 0  # 时间点获取下标
        self.slow_i = 0
        self.veryslow_i = 0
        self.change_num_wh = list()  # 每次gap插入的新库数量
        self.change_num_car = list()  # 每次gap插入的新车数量
        self.change_type = list()  # 插入类型：有影响、无影响、混合
        self.wh_i = 0
        self.car_i = 0
        self.type_i = 0

        self.static_point = list()  # park、loading属性更新时间点，按比例从gap次数随机
        self.static_type = list()  # 修改类型：park、loading、分别与gap混合
        self.static_num = list()  # 每次更新的仓库或者货物数
        self.static_loading = list()  # 更新货物时的修改值
        self.static_park = list()  # 更新仓库时的停车数值
        self.static_time = list()  # 更新的时间
        self.s_type_i = 0
        self.s_num_i = 0
        self.s_loading_i = 0
        self.s_park_i = 0
        self.s_time_i = 0

        self.p_time = 0  # 用作当天插入时间范围结束后，只更新数据的时间间隔
                         # 实现方式是每次进到这里判断里就+1，去可能时间增量值的模

        self.tablea_num = list()  # 每次董涛更新时决定这个无关表需要更新条数据
        self.tablea_time = list()  # 每条数据的时间间隔
        self.ta_num_i = 0
        self.ta_time_i = 0
        self.tcol_num = list()  # 每次取三个值，表示在三个相关表中分别增加的一个无关项的变化次数
        self.tcol_time = list()
        self.tc_num_i = 0
        self.tc_time_i = 0

        self.create_freqfile()    # 时间点三个file，快、慢、很慢
        self.create_changefile()  # 新车插入三个file，插入库量，插入车数、插入类型（有影响、无影响、混合）
        self.create_staticfile()  # 静态更新的六个file，更新时间点，更新类型（park、loading、分别混合gap），
                                  # 每次更新的仓库或者货物数，更新的时间值，更新的停车值，更新值的时间
        self.create_tableafile()  # 增加的与场景无关的表数据，两个file，更新个数，时间间隔。多配点0的更新个数
                                  # 类似方式新增两个文件，用于相关表中的新增无关项

        self.create_queuing_table()

    #初始化负载对象操作一次  创建一个空表，一个无关数据表
    def create_queuing_table(self):
        if self.basedata == 1:
            for ip in ['106.75.233.244', '106.75.244.49']:
                db = pymysql.connect(host=ip, user='root', password='huangss123',
                                     database='accuqueuedata', charset='utf8')
                cursor = db.cursor()

                # 排队表
                sql = "DROP TABLE IF EXISTS queuing_info;"
                cursor.execute(sql)
                db.commit()
                cursor.execute(sql_create_queuing_info)
                db.commit()

                # 无关表
                sql = "DROP TABLE IF EXISTS tablea;"
                cursor.execute(sql)
                db.commit()
                cursor.execute(sql_create_tablea)
                db.commit()
                # 插入一条起始数据
                cursor.execute(sql_insert_tablea, (10, 10, 10, 10, '2021-01-01 05:59:50'))
                db.commit()

                cursor.close()
                db.close()

    def create_freqfile(self):
        # 创建基础的更改频率时间文件
        if self.basedata == 1:
            freq_fast = [random.randint(10, 180) for _ in range(10000)]
            freq_slow = [random.randint(300, 480) for _ in range(5000)]
            freq_veryslow = [random.randint(600, 1200) for _ in range(2000)]
            with open("/root/dataset_accu/queue/freq_fast.txt", 'w') as file:
                file.write(str(freq_fast))
            with open("/root/dataset_accu/queue/freq_slow.txt", 'w') as file:
                file.write(str(freq_slow))
            with open("/root/dataset_accu/queue/freq_veryslow.txt", 'w') as file:
                file.write(str(freq_veryslow))
        # 读取
        with open("/root/dataset_accu/queue/freq_fast.txt", 'r') as file:
            self.freq_fast = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/freq_slow.txt", 'r') as file:
            self.freq_slow = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/freq_veryslow.txt", 'r') as file:
            self.freq_veryslow = ast.literal_eval(file.readline().rstrip())

        # print(f"freq_fast: {self.freq_fast}")
        # print(f"freq_slow: {self.freq_slow}")
        # print(f"freq_veryslow: {self.freq_veryslow}")

    def create_changefile(self):
        # 操作同上，只是file用作不用，分开写好区分
        if self.basedata == 1:
            change_num_wh = [random.randint(1, 5) for _ in range(6000)]
            change_num_car = [random.randint(2, 8) for _ in range(6000)]  # 下限不能是1，否则gap无效
            change_type = [random.randint(1, 3) for _ in range(6000)]  #
            with open("/root/dataset_accu/queue/change_num_wh.txt", 'w') as file:
                file.write(str(change_num_wh))
            with open("/root/dataset_accu/queue/change_num_car.txt", 'w') as file:
                file.write(str(change_num_car))
            with open("/root/dataset_accu/queue/change_type.txt", 'w') as file:
                file.write(str(change_type))
        with open("/root/dataset_accu/queue/change_num_wh.txt", 'r') as file:
            self.change_num_wh = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/change_num_car.txt", 'r') as file:
            self.change_num_car = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/change_type.txt", 'r') as file:
            self.change_type = ast.literal_eval(file.readline().rstrip())

        # print(f"change_num_wh: {self.change_num_wh}")
        # print(f"change_num_car: {self.change_num_car}")
        # print(f"change_type: {self.change_type}")

    def create_staticfile(self):
        if self.basedata == 1:
            gap = 2000  # 注意设置这个值的时候，分别是我们的 收集+测试 的，别设置少了
            #  15:1 去修改park_cout和loading_time
            static_point = random.sample(range(1, gap+1), int(gap / 15))     # 随机数不包括尾
            static_point.sort()
            # 改park 1，改loading 2，混合插入新车3,4
            static_type = [random.randint(1, 4) for _ in range(int(gap/15)+4)]   # 按比例15:1抽，免得移出加个4
            # 改动仓库或者货物个数，只在1,2中随机
            static_num = [random.randint(1, 2) for _ in range(1500)]
            # 改动时间
            static_loading = [random.randint(2, 15) for _ in range(1500)]
            # 改动停车数
            static_park = [random.randint(0, 10) for _ in range(1500)]
            # 做操作的更新时间
            static_time = [random.randint(5, 10) for _ in range(1500)]
            with open("/root/dataset_accu/queue/static_point.txt", 'w') as file:
                file.write(str(static_point))
            with open("/root/dataset_accu/queue/static_type.txt", 'w') as file:
                file.write(str(static_type))
            with open("/root/dataset_accu/queue/static_num.txt", 'w') as file:
                file.write(str(static_num))
            with open("/root/dataset_accu/queue/static_loading.txt", 'w') as file:
                file.write(str(static_loading))
            with open("/root/dataset_accu/queue/static_park.txt", 'w') as file:
                file.write(str(static_park))
            with open("/root/dataset_accu/queue/static_time.txt", 'w') as file:
                file.write(str(static_time))
        with open("/root/dataset_accu/queue/static_point.txt", 'r') as file:
            self.static_point = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/static_type.txt", 'r') as file:
            self.static_type = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/static_num.txt", 'r') as file:
            self.static_num = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/static_loading.txt", 'r') as file:
            self.static_loading = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/static_park.txt", 'r') as file:
            self.static_park = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/static_time.txt", 'r') as file:
            self.static_time = ast.literal_eval(file.readline().rstrip())

    def create_tableafile(self):
        # 增加的与场景无关的表数据，两个file，更新个数，时间间隔。多配点0的更新个数
        # 用类似的做法更新三个表表中新增的无关项
        if self.basedata == 1:
            # 下面的小数用来调整0出现的概率（random.random() 来生成一个 0 到 1 之间的随机小数）
            # 注意，这里的range数值改了，上面关于static出现频率文件也需要跟着修改。static_point和static_type
            tablea_num = [0 if random.random() < 0.2 else random.randint(5, 20) for _ in range(2000)]  # 一gap一次
            # tablea_time = [random.uniform(0, 8) for _ in range(10000)]
            tablea_time = [random.randint(10, 30) for _ in range(60000)]

            tcol_num = [0 if random.random() < 0.1 else random.randint(5, 20) for _ in range(12000)]  # 一gap三次
            # tcol_time = [random.uniform(0, 6) for _ in range(20000)]
            tcol_time = [random.randint(10, 30) for _ in range(120000)]

            with open("/root/dataset_accu/queue/tablea_num.txt", 'w') as file:
                file.write(str(tablea_num))
            with open("/root/dataset_accu/queue/tablea_time.txt", 'w') as file:
                file.write(str(tablea_time))
            with open("/root/dataset_accu/queue/tcol_num.txt", 'w') as file:
                file.write(str(tcol_num))
            with open("/root/dataset_accu/queue/tcol_time.txt", 'w') as file:
                file.write(str(tcol_time))
        with open("/root/dataset_accu/queue/tablea_num.txt", 'r') as file:
            self.tablea_num = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/tablea_time.txt", 'r') as file:
            self.tablea_time = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/tcol_num.txt", 'r') as file:
            self.tcol_num = ast.literal_eval(file.readline().rstrip())
        with open("/root/dataset_accu/queue/tcol_time.txt", 'r') as file:
            self.tcol_time = ast.literal_eval(file.readline().rstrip())

    # 获取上一条记录的开始时间
    def get_last_start_time(self):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        sql = "SELECT start_time FROM queuing_info ORDER BY recordid DESC LIMIT 1;"
        cursor.execute(sql)
        last_start_time = cursor.fetchone()[0]
        cursor.close()
        db.commit()

        return last_start_time

    # 生成累积
    # 不考虑前序排位信息，只是根据指定生成指定仓库的排队排队车，车的货物是按仓库kind循环的
    # 参数： 仓库编号，仓库下车车辆数，到达时间（如果为空就接上一条记录的时间，每辆到达的间隔
    def create_queuedata(self, warehouse_num, num, come_time, sec):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()

        # 初始化，userid是 U000，按self.user_num中起始的顺序
        # 根据指定仓库和排队数量，物品循环，然后打乱放进去再加算时间
        queue_df = pd.DataFrame(columns=['user_code', 'cargo_name', 'warehouse_name'])
        # warehouse_num = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        # warehouse_num = [6, 21]
        # num = 15
        cargo_all = list()  # 总的初始化，包含指定的所有的仓库
        warehouse_all = list()
        for wid in warehouse_num:
            sql = "select warehouse_name, cargo_kind from warehouse_info where warehouseid = %s;"
            cursor.execute(sql, (wid,))
            r = cursor.fetchone()
            kind = ast.literal_eval(r[1])
            # 根据当前仓库和货物列表生成的list。货物循环放置
            ware_list = [r[0] for _ in range(num)]
            kind_list = list()
            for i in range(num):
                kind_list.append(kind[i % len(kind)])
            # print(ware_list)
            # print(kind_list)

            cargo_all.extend(kind_list)
            warehouse_all.extend(ware_list)

        # 补充usercode
        user_all = list()
        start = self.user_num
        end = self.user_num + len(warehouse_num) * num
        self.user_num = end
        for i in range(start, end):
            user_all.append(f"U{i:06}")

        # 初始化数据组合并打乱
        queue_df['user_code'] = user_all
        queue_df['cargo_name'] = cargo_all
        queue_df['warehouse_name'] = warehouse_all
        # print(queue_df)
        # 打乱
        shuffled_df = queue_df.sample(frac=1, random_state=33)
        # print(shuffled_df)
        # 重现
        # original_order_df = queue_df.sample(frac=1, random_state=33)
        # print(original_order_df)

        # 直接将开始时间补充到数据库（因为数据是按时间顺序到的）
        start_time_all = list()
        if come_time:   # 不为空
            current_time = datetime.strptime(come_time, '%Y-%m-%d %H:%M:%S')  # 解析起始时间字符串
        else:
            current_time = self.get_last_start_time()
        for _ in range(len(user_all)):
            current_time += timedelta(seconds=sec)  # 增加指定的时间间隔
            start_time_all.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))  # 格式化当前时间并添加到列表中
        shuffled_df['start_time'] = start_time_all
        # print(shuffled_df)

        # 插入数据
        # insert_warehouse = list()
        for _, r in shuffled_df.iterrows():
            sql = "INSERT INTO queuing_info (user_code, cargo_name, warehouse_name, start_time) VALUES(%s, %s, %s, %s);"
            params = (r['user_code'], r['cargo_name'], r['warehouse_name'], r['start_time'])
            filled_sql = cursor.mogrify(sql, params)
            # print(filled_sql)
            cursor.execute(sql, params)
            # insert_warehouse.append(r['warehouse_name'])
        db.commit()

        ###### 生成少量数据，为了测试其他模块功能，后期可注释 ######
        # 这里格外添加保存在excel，是为了生成小一点的数据，手动进行修改再上传，测试实现的功能是否正确
        # 这一段后期不用了，可以注释掉
        # sql = "SELECT * FROM queuing_info;"
        # queuing_df = pd.read_sql_query(sql, db)  # 这个函数直接将查询结果生成dataframe
        # queuing_df.to_excel('/root/dataset_accu/queuing_info.xlsx', index=False, encoding='utf-8')
        #############

        cursor.close()
        db.close()

        # return insert_warehouse  # 用于负载的时候判断添加数据的仓库名。已经去重过。  注意：其实在自己插入数据的时候就指定了


    # 主库同步到备库，这里直接查询然后插入过去即可，包括库下的三个表
    def queuing_syn(self):
        primaryDB = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                    database='accuqueuedata', charset='utf8')
        secondaryDB = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                      database='accuqueuedata', charset='utf8')
        cursor_P = primaryDB.cursor()
        cursor_S = secondaryDB.cursor()

        # 1. 备库表均删除并重建
        sql = "DROP TABLE IF EXISTS queuing_info;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        cursor_S.execute(sql_create_queuing_info)
        secondaryDB.commit()
        sql = "DROP TABLE IF EXISTS warehouse_info;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        cursor_S.execute(sql_create_warehouse_info)
        secondaryDB.commit()
        sql = "DROP TABLE IF EXISTS cargo_info;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        cursor_S.execute(sql_create_cargo_info)
        secondaryDB.commit()
        sql = "DROP TABLE IF EXISTS tablea;"
        cursor_S.execute(sql)
        secondaryDB.commit()
        cursor_S.execute(sql_create_tablea)
        secondaryDB.commit()

        # 2. 主库的表sql查询全表后插入到备库
        sql = "SELECT * FROM queuing_info;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            values = (row['recordid'], row['user_code'], row['cargo_name'], row['warehouse_name'],
                      row['start_time'], row['entry_time'], row['entry_whouse_time'], row['finish_time'],
                      row['qt'])
            # NaN不能直接当做参数值用到sql中，需要转为NULL
            new_vaules = tuple(None if x != x else x for x in values)
            cursor_S.execute(sql_insert_queuing_info, new_vaules)
        sql = "SELECT * FROM warehouse_info;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            values = (row['warehouseid'], row['warehouse_name'], row['park_count'], row['cargo_kind'],
                      row['wt'])
            cursor_S.execute(sql_insert_warehouse_info, values)
        sql = "SELECT * FROM cargo_info;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            values = (row['cargoid'], row['cargo_name'], row['loading_time'],
                      row['ct'])
            cursor_S.execute(sql_insert_cargo_info, values)
        sql = "SELECT * FROM tablea;"
        queue_df = pd.read_sql_query(sql, primaryDB)
        for index, row in queue_df.iterrows():
            values = (row['t1'], row['t2'], row['t3'], row['t4'], row['t5'])
            cursor_S.execute(sql_insert_tablea, values)

        secondaryDB.commit()
        cursor_P.close()
        cursor_S.close()
        primaryDB.close()
        secondaryDB.close()

    # 更新“加新车”负载时需要只知道被使用的仓库。如果不是请求者的仓库被修改，改了没有效果，此时应该是有空车位
    def get_used_warehouse(self):
        # 只查主库queuing_info
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        sql = "SELECT DISTINCT warehouse_name FROM queuing_info WHERE entry_time IS NULL;"
        cursor.execute(sql)
        result = cursor.fetchall()
        has_warehouse = [row[0] for row in result]  # name
        cursor.close()
        db.close()
        return has_warehouse

    # （1）插入新车辆，
    #  → 和之前仓库不重复，可能没有请求
    #  → 重复，很大程度要等待
    # 参数1：插入时间，是时间类型的
    # 参数2、3：有影响的插入  list: len为仓库个数，值是每个库要增加的车数
    #                     如果list超过“之前用的仓库”，则循环
    #                     为 []，表示此时不插入会有影响查询。注意此时只是不新增了，可能之前的数据状态还有是要查询的。查新在但是预测没误差
    ### 已使用库，分别每库插入has_wh车数，插入时间has_time。len超过将循环已使用库
    # 参数4：无影响, list：len插入的新仓库数，值是对应时间间隔
    #                插入数量不会超过park。 0时不插入，1-5插1,6-8插2,9-10插3
    ### 未使用库，新增车数len，值为时间间隔，默认车数
    def update_newtrucks(self, insert_time, has_wh, has_time, no_wh):
        # 之前用到的仓库，是name
        used_wh = self.get_used_warehouse()
        used_wh_id = list()
        # 新增数据，这些主备差异可能导致预测误差
        if has_wh:  # 要新增。根据之前有用到的仓库来添加
            if used_wh: # 如果没有已使用库，无法进行更新
                for wh in used_wh:
                    used_wh_id.append(self.all_warehouse.index(wh) + 1)  # 表中仓库id对应的是从1开始的
                for i in range(len(has_wh)):
                    self.create_queuedata([used_wh_id[i % len(used_wh_id)]], has_wh[i],
                                              insert_time.strftime('%Y-%m-%d %H:%M:%S'), has_time[i])
                    insert_time += timedelta(seconds=has_wh[i] * has_time[i])

        # 增加新数据，是新的数据库，通常没影响，只要数量不大于仓库数
        if no_wh:  # 如果为0，表示没有添加仓库id数
            # 1）过滤按顺序得到no_wh个未使用过的仓库id及其park数
            all_warehouse_id = list(range(1, 51))  # 总仓库是1-50编号
            unused_all_warehouse_id = [i for i in all_warehouse_id if i not in used_wh_id]  # 去掉已使用的仓库
            # 如果增新库大于剩余的，那么截断
            if len(unused_all_warehouse_id) <= len(no_wh):
                new_warehouse_id = unused_all_warehouse_id
            else:
                new_warehouse_id = unused_all_warehouse_id[: len(no_wh)]
            # 2）将使用的仓库对应的park
            db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                 database='accuqueuedata', charset='utf8')
            cursor = db.cursor()
            new_warehouse_park = list()
            sql = "SELECT park_count FROM warehouse_info WHERE warehouseid IN %s;"
            cursor.execute(sql, (tuple(new_warehouse_id),))
            result = cursor.fetchall()
            cursor.close()
            db.close()
            for r in result:
                new_warehouse_park.append(r[0])
            # 3) 插入仓库
            for wh_i in range(len(new_warehouse_id)):
                if new_warehouse_park[wh_i] == 0:
                    pass
                elif new_warehouse_park[wh_i] >= 1 and new_warehouse_park[wh_i] <= 5:
                    self.create_queuedata([new_warehouse_id[wh_i]], 1,
                                              insert_time.strftime('%Y-%m-%d %H:%M:%S'), no_wh[wh_i])
                    insert_time += timedelta(seconds=1 * no_wh[wh_i])
                elif new_warehouse_park[wh_i] >= 4 and new_warehouse_park[wh_i] <= 7:
                    self.create_queuedata([new_warehouse_id[wh_i]], 2,
                                              insert_time.strftime('%Y-%m-%d %H:%M:%S'), no_wh[wh_i])
                    insert_time += timedelta(seconds=2 * no_wh[wh_i])
                else:  # new_warehouse_park[wh_i] >= 9 or new_warehouse_park[wh_i] <= 10:
                    self.create_queuedata([new_warehouse_id[wh_i]], 3,
                                              insert_time.strftime('%Y-%m-%d %H:%M:%S'), no_wh[wh_i])
                    insert_time += timedelta(seconds=3 * no_wh[wh_i])

        return insert_time

    # (2) 修改loading（货物表的加载时间，不要改太大，一般2-15分钟）
    #  → 前车有的货物被改，有影响
    #  → 改没有的货物
    # 参数1 更改时，同时要更改版本表。修改时间为传入时间的2秒后。
    # 参数2,3： 表示要修改的货物加载时间list。如果超出只修改有的最大数量
    ### 存在货物，将修改的加载时间，修改时间，超过截断。修改值与原来不同，都则除2取整
    # 参数:4,5： 未使用货物的loading修改，对预测不影响，修改len(no_cargo)个货物，对应list值为修改值
    ### 新货物，同上
    def update_loading(self, update_time, has_cargo, has_time, no_cargo, no_time):
        #  前车货物---
        # 只筛选出会有请求的仓库，对这些库的前序车辆货物进行收集，然后去掉最新以为车的货物。再去重
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        # 请求着仓库，即entry为null的
        used_warehouse = tuple(self.get_used_warehouse())
        # print(used_warehouse)
        if used_warehouse:   #没有用户请求，那么就没有仓库对应的货物，则修改有影响的操作不能进行
            # 同请求者仓库的货物，前车中的，即finish为null
            sql = "SELECT cargo_name FROM queuing_info WHERE warehouse_name IN %s AND finish_time IS NULL;"
            # print(cursor.mogrify(sql, (used_warehouse,)))
            cursor.execute(sql, (used_warehouse,))
            result = cursor.fetchall()
            fornt_cargo_all = [i[0] for i in result]
            # 去掉不同仓库的最后一位访问者的货物。这里entry是null表示只有可请求车辆的货物信息了
            sql = "SELECT cargo_name " \
                  "FROM ( SELECT cargo_name, (SELECT COUNT(*) FROM queuing_info t2 " \
                  "WHERE t2.warehouse_name = t1.warehouse_name AND t2.start_time >= t1.start_time) " \
                  "AS row_number FROM queuing_info t1 WHERE finish_time IS NULL AND entry_time IS NULL) " \
                  "AS RankedData WHERE row_number = 1;"
            # SELECT
            #     recordid,
            #     warehouse_name
            # FROM (
            #     SELECT
            #         recordid,
            #         warehouse_name,
            #         (SELECT COUNT(*)
            #         FROM queuing_info t2
            #         WHERE t2.warehouse_name = t1.warehouse_name AND t2.start_time >= t1.start_time
            #         ) AS row_number
            #     FROM queuing_info t1
            #     WHERE finish_time IS NULL AND entry_time IS NULL
            # ) AS RankedData
            # WHERE row_number = 1;
            cursor.execute(sql)
            result = cursor.fetchall()
            for r in result:
                fornt_cargo_all.remove(r[0])
            front_cargo = list(set(fornt_cargo_all))
            # print(f"前车货物：{front_cargo}")
            not_in_front_cargo = [i for i in self.all_cargo if i not in front_cargo]  # 从全集货物中筛选出来的未使用的货物名
            # print(f"所有新货物：{not_in_front_cargo}")
            # 修改数据，还有修改表. list的内容是改动时长，不建议太长
            if has_cargo:
                if front_cargo:  #没有被使用货物，这里无法更新
                    # 更改数超过被使用货物数，则只修改存在的货物量个
                    if len(has_cargo) <= len(front_cargo):
                        new_has_cargo = front_cargo[: len(has_cargo)]
                    else:
                        new_has_cargo = front_cargo
                    # 保证修改后不是原来的值。这里需要根据cargo_name查询原的值，如果相同则除2取整
                    new_has_cargo_loading = list()
                    for c_name in new_has_cargo:
                        sql = "SELECT loading_time FROM cargo_info WHERE cargo_name = %s;"
                        cursor.execute(sql, (c_name,))
                        new_has_cargo_loading.append(cursor.fetchone()[0])
                    has_cargo_new0 = [has_cargo[i] // 2 if has_cargo[i] == new_has_cargo_loading[i]
                                     else has_cargo[i] for i in range(len(new_has_cargo_loading))]
                    # 为0的需要改为2
                    has_cargo_new = [2 if has_cargo_new0[i] == 0 else has_cargo_new0[i] for i in range(len(has_cargo_new0))]

                    # 开始修改
                    for c in range(len(new_has_cargo)):
                        sql = "UPDATE cargo_info SET loading_time = %s WHERE cargo_name = %s;"
                        cursor.execute(sql, (has_cargo_new[c], new_has_cargo[c]))
                        # 还有版本表的更新
                        sql = "UPDATE accuqueueversion.vtime_loading_time SET version = %s;"
                        update_time += timedelta(seconds=has_time[c])
                        cursor.execute(sql, (update_time,))
                        sql = "UPDATE accuqueueversion.vnum_loading_time SET version = version + 1;"
                        cursor.execute(sql)
                        db.commit()
        if no_cargo:
            # 如果没被使用的货物，全部可被使用
            if not used_warehouse:
                not_in_front_cargo = self.all_cargo
            # 新货物数没有参数要求的那么多，则增加有的数量
            if len(not_in_front_cargo) <= len(no_cargo):
                new_cargo_name = not_in_front_cargo
            else:
                new_cargo_name = not_in_front_cargo[: len(no_cargo)]
            for c in range(len(new_cargo_name)):
                sql = "UPDATE cargo_info SET loading_time = %s WHERE cargo_name = %s;"
                cursor.execute(sql, (no_cargo[c], new_cargo_name[c]))
                # 还有版本表的更新
                sql = "UPDATE accuqueueversion.vtime_loading_time SET version = %s ;"
                update_time += timedelta(seconds=no_time[c])
                cursor.execute(sql, (update_time,))
                sql = "UPDATE accuqueueversion.vnum_loading_time SET version = version + 1 ;"
                cursor.execute(sql)
                db.commit()

        cursor.close()
        db.close()
        return update_time

    # (3) 修改park（仓库表的停车数，值为0-10，改为和之前不一样，一样的话除2取整）
    #  → 请求者是这个仓库，有影响  (改动的值，改动的时间
    #  → 请求者不是这个仓库
    def update_park(self, update_time, has_park, has_time, no_park, no_time):
        # 当前可请求用户的目标仓库
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        used_wh = self.get_used_warehouse()
        not_used_wh = [i for i in self.all_warehouse if i not in used_wh]
        if has_park:
            if used_wh:    #没有被使用仓库，这里无法更新
                # 需要更新库超过使用的，只更新被使用个仓库数
                if len(has_park) <= len(used_wh):
                    new_used_wh = used_wh[: len(has_park)]
                else:
                    new_used_wh = used_wh
                # 保证修改后不是原来的值。这里需要根据wh_name查询原的值，如果相同则除2取整
                new_used_wh_park = list()
                for wh_name in new_used_wh:
                    sql = "SELECT park_count FROM warehouse_info WHERE warehouse_name = %s;"
                    cursor.execute(sql, (wh_name, ))
                    new_used_wh_park.append(cursor.fetchone()[0])
                has_park_new = [has_park[i] // 2 if has_park[i] == new_used_wh_park[i]
                                else has_park[i] for i in range(len(new_used_wh_park))]
                # 开始修改
                for c in range(len(new_used_wh)):
                    sql = "UPDATE warehouse_info SET park_count = %s WHERE warehouse_name = %s;"
                    cursor.execute(sql, (has_park_new[c], new_used_wh[c]))
                    # 还有版本表的更新
                    sql = "UPDATE accuqueueversion.vtime_park_count SET version = %s;"
                    update_time += timedelta(seconds=has_time[c])
                    cursor.execute(sql, (update_time,))
                    sql = "UPDATE accuqueueversion.vnum_park_count SET version = version + 1;"
                    cursor.execute(sql)
                    db.commit()
        if no_park:
            # 新库没有参数要求的那么多，则截断
            if len(not_used_wh) <= len(no_park):
                new_cargo_name = not_used_wh
            else:
                new_cargo_name = not_used_wh[: len(no_park)]
            for c in range(len(new_cargo_name)):
                sql = "UPDATE warehouse_info SET park_count = %s WHERE warehouse_name = %s;"
                cursor.execute(sql, (no_park[c], new_cargo_name[c]))
                # 还有版本表的更新
                sql = "UPDATE accuqueueversion.vtime_park_count SET version = %s ;"
                update_time += timedelta(seconds=no_time[c])
                cursor.execute(sql, (update_time,))
                sql = "UPDATE accuqueueversion.vnum_park_count SET version = version + 1 ;"
                cursor.execute(sql)
                db.commit()

        cursor.close()
        db.close()
        return update_time

    # 存在只更改静态且需要有影响的操作，可此如果没有请求，改了也没用，所以此处手动添加一些可发送请求的用户
    def add_request_user(self, update_time):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        sql = "SELECT recordid, user_code, cargo_name, warehouse_name, start_time " \
              "FROM queuing_info WHERE entry_time IS NULL;"
        cursor.execute(sql)
        result = cursor.fetchall()
        if result:
            # 取park在3及以下，下面新增数要至少为5
            sql = "SELECT warehouseid FROM warehouse_info WHERE park_count < 4 limit 2;"
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            db.close()
            if result: # 没有少量park的库可以添加，那就直接选定一个库，加多一点新车
                self.create_queuedata([3], 14, '', 5)
                add_time = 5 * 14
            else:
                add_time = 0
                for r in result:
                    self.create_queuedata([r[0]], 4, '', 5)  # 仓库编号，车辆数，''，时间间隔
                    add_time += (4*5)
            update_time += timedelta(seconds=add_time)
        return update_time

    def view_classdata(self):
        print(f"更新时间间隔快、慢、极慢：{self.fast_i}, {self.slow_i}, {self.veryslow_i}")
        print(f"新输入相关：库数量 {self.wh_i}, 每库下新车数 {self.car_i}, 有影响无影响混合{self.type_i}")
        print(f"静态修改类型 {self.s_type_i}, 修改数量 {self.s_num_i}, 更新时间 {self.s_time_i}")
        print(f"静态修改值 加载值 {self.s_loading_i}, 停车值 {self.s_park_i}")
        print(f"时间范围外的更新次数 {self.p_time}")
        print(f"无关数据表每次更新次数 {self.ta_num_i}, 每行更新间隔 {self.ta_time_i}")
        print(f"无关数据项每次更新次数 {self.tc_num_i}, 每次更新间隔 {self.tc_time_i}")


    def dynmaicload(self, update_time, statici):
        # print(f"这一轮开始时的各文件index（也就是一轮的index")
        # self.view_classdata()

        # 无关数据表的更新
        need_insert_rownum = self.tablea_num[self.ta_num_i]
        # 这个表的更新时间可以时其他的同步，所以记录最后更新完的时间，在负载结束的时候需要判断最后返回的时间
        tablea_updae_time = update_time
        if need_insert_rownum:
            # print(f"tables改了{need_insert_rownum}行")
            db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                 database='accuqueuedata', charset='utf8')
            cursor = db.cursor()
            has_time = self.tablea_time[self.ta_time_i: self.ta_time_i+need_insert_rownum]
            for ht in has_time:
                t1234_v = np.random.randint(low=5, high=7000, size=4) + np.random.rand(4)
                t5_v = tablea_updae_time + timedelta(seconds=ht)
                # print(f"时间分别是{t5_v}")
                params = (t1234_v[0], t1234_v[1], t1234_v[2], t1234_v[3], t5_v)
                filled_sql = cursor.mogrify(sql_insert_tablea, params)
                # print(filled_sql)
                cursor.execute(sql_insert_tablea, params)
                db.commit()
                tablea_updae_time = t5_v
            cursor.close()
            db.close()
        self.ta_num_i += 1
        self.ta_time_i += need_insert_rownum
        # print(f"更新完无关数据表后的时间{tablea_updae_time}")

        # 相关表中的无关数据项的更新
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        need1 = self.tcol_num[self.tc_num_i]  # 三个相关表中每个无关数据项更新次数
        need2 = self.tcol_num[self.tc_num_i+1]
        need3 = self.tcol_num[self.tc_num_i+2]
        self.tc_num_i += 3
        tcol_updae_time = update_time  # 注意，同上，这里的执行也可以是并行的，所以需要额外记时间，最后进行比较
        # print(f"针对q、w、c的修改次数分别为：{need1}，{need2}，{need3}")
        if need1:
            has_time = self.tcol_time[self.tc_time_i: self.tc_time_i+need1]
            self.tc_time_i += need1
            # 在queuing_info已存在行中，随机行id进行修改.
            # 这里可以简写，只随机一个，然后把对应值改为最后累积的时间值，版本num也是累积的个数即可
            tcol_updae_time += timedelta(seconds=sum(has_time))
            sql = "update queuing_info set qt = %s where recordid=10;"
            cursor.execute(sql, (tcol_updae_time, ))
            # print(f"时间分别是{tcol_updae_time}")
        if need2:
            has_time = self.tcol_time[self.tc_time_i: self.tc_time_i+need2]
            self.tc_time_i += need2
            tcol_updae_time += timedelta(seconds=sum(has_time))
            sql = "update warehouse_info set wt = %s where warehouseid=10;"
            cursor.execute(sql, (tcol_updae_time, ))
            # print(f"时间分别是{tcol_updae_time}")
        if need3:
            has_time = self.tcol_time[self.tc_time_i: self.tc_time_i+need3]
            self.tc_time_i += need3
            tcol_updae_time += timedelta(seconds=sum(has_time))
            sql = "update cargo_info set ct = %s where cargoid=10;"
            cursor.execute(sql, (tcol_updae_time, ))
            # print(f"时间分别是{tcol_updae_time}")
        # 最后一起更新版本表信息
        if need1 or need2 or need3:
            sql = "update accuqueueversion.vnum_tcol set qn = qn + %s, wn = wn + %s, cn = cn + %s;"
            cursor.execute(sql, (need1, need2, need3))
        db.commit()
        cursor.close()
        db.close()
        # print(f"更新完无关项后的时间{tcol_updae_time}")

        # 静态特征修改在gap次数时间点
        if statici in self.static_point:
            now_type = self.static_type[self.s_type_i]
            has_time = self.static_time[self.s_time_i: (self.s_time_i + self.static_num[self.s_num_i])]
            if now_type == 1 or now_type == 3:
                has_num = self.static_park[self.s_park_i: (self.s_park_i + self.static_num[self.s_num_i])]
                update_time = self.update_park(update_time, has_num, has_time, [], [])
                self.s_park_i += self.static_num[self.s_num_i]
            elif now_type == 2 or now_type == 4:
                has_num = self.static_loading[self.s_loading_i: (self.s_loading_i + self.static_num[self.s_num_i])]
                update_time = self.update_loading(update_time, has_num, has_time, [], [])
                self.s_loading_i += self.static_num[self.s_num_i]
            self.s_num_i += 1
            self.s_time_i += self.static_num[self.s_num_i]
            self.s_type_i += 1
            if now_type == 1 or now_type == 2:  # 如果此时没有请求，那么改了静态也没用，所以自动加一点新车数
                update_time = self.add_request_user(update_time)
                # 因为不相关表的更新时可以同步于排队场景的更新的，所以要判断，将大的返回
                all_possibe_time = [tablea_updae_time, tcol_updae_time, update_time]
                return max(all_possibe_time)

        # 一下是gap更新
        t6 = datetime.strptime('06:00:00', '%H:%M:%S').time()
        t9 = datetime.strptime('09:00:00', '%H:%M:%S').time()
        t11 = datetime.strptime('11:00:00', '%H:%M:%S').time()
        t12 = datetime.strptime('12:00:00', '%H:%M:%S').time()
        t13 = datetime.strptime('13:00:00', '%H:%M:%S').time()

        flag = ''
        # 只和time部分对比，不要日期
        time_part = update_time.time()
        if (time_part > t6 and time_part < t9) \
                or (time_part > t12 and time_part < t13):
            flag = 'f'
            # print("这轮是fast")
        elif (time_part > t9 and time_part < t11):
            flag = 's'
            # print("这轮是slow")
        elif time_part > t11 and time_part < t12:
            flag = 'v'
            # print("这轮是veryslow")
        else:
            # 不在时间内，就表明是晚上休息时间，现在只将时间往后更新就好，直到没有请求
            # 这部分时间取常见会，且之前收集比较缺的
            possible_delta = [80, 90, 110, 120]
            p_i = self.p_time % len(possible_delta)
            self.p_time += 1
            possible_time = update_time + timedelta(minutes=possible_delta[p_i])
            # 不能到下一天了，否则当天最大
            if possible_time.date() < (update_time.date() + timedelta(days=1)):
                all_possibe_time = [tablea_updae_time, tcol_updae_time, update_time]
                return max(all_possibe_time)
            else:
                return update_time.replace(hour=23, minute=59, second=0)

        # 要更新的时间点
        if flag == 'f':
            freq_list = self.freq_fast
            freq_i = self.fast_i
        elif flag == 's':
            freq_list = self.freq_slow
            freq_i = self.slow_i
        else:
            freq_list = self.freq_veryslow
            freq_i = self.veryslow_i

        # 插入类型：有影响1，无影响2，混合3
        # update_newtrucks(insert_time, has_wh, has_time, no_wh) 存在仓库库下新增数量、时间。 新增仓库的时间，库下车数是默认的
        if self.change_type[self.type_i] == 1:
            has_wh = self.change_num_car[self.car_i: (self.car_i + self.change_num_wh[self.wh_i])]
            has_time = freq_list[freq_i: (freq_i + self.change_num_wh[self.wh_i])]
            self.car_i += self.change_num_wh[self.wh_i]
            # print("更新类型是：有影响")
            # print(f"更新仓库对应车数：{has_wh}，对应修改时间：{has_time}")
            update_time = self.update_newtrucks(update_time, has_wh, has_time, [])
            # print(f"此次更新后的时间：{update_time}")
        elif self.change_type[self.type_i] == 2:
            no_wh = freq_list[freq_i: (freq_i + self.change_num_wh[self.wh_i])]
            # print("更新类型是：无影响")
            # print(f"更新仓库对应修改时间：{no_wh}")
            update_time = self.update_newtrucks(update_time, [], [], no_wh)
            # print(f"此次更新后的时间：{update_time}")
        else:
            has_wh = self.change_num_car[self.car_i: (self.car_i + self.change_num_wh[self.wh_i])]
            has_time = freq_list[freq_i: (freq_i + self.change_num_wh[self.wh_i])]
            # 因为下标需要再次使用，所以这里也要更新
            self.car_i += self.change_num_wh[self.wh_i]
            if flag == 'f':
                self.fast_i += self.change_num_wh[self.wh_i]
            elif flag == 's':
                self.slow_i += self.change_num_wh[self.wh_i]
            else:
                self.veryslow_i += self.change_num_wh[self.wh_i]
            freq_i += self.change_num_wh[self.wh_i]
            self.wh_i += 1
            no_wh = freq_list[freq_i: (freq_i + self.change_num_wh[self.wh_i])]
            # print("更新类型是：有影响+无影响")
            # print(f"有影响，更新仓库对应车数：{has_wh}，对应修改时间：{has_time}")
            # print(f"无影响，更新仓库对应修改时间：{no_wh}")
            update_time = self.update_newtrucks(update_time, has_wh, has_time, no_wh)
            # print(f"此次更新后的时间：{update_time}")

        # 下次使用
        self.type_i += 1
        if flag == 'f':
            self.fast_i += self.change_num_wh[self.wh_i]
        elif flag == 's':
            self.slow_i += self.change_num_wh[self.wh_i]
        else:
            self.veryslow_i += self.change_num_wh[self.wh_i]
        self.wh_i += 1

        # print(f"fast更新时间：{self.fast_i}")
        # print(f"slow更新时间：{self.slow_i}")
        # print(f"veryslow更新时间：{self.veryslow_i}")
        # print(f"要更新仓库数：{self.wh_i}")
        # print(f"仓库下的新车到来数：{self.car_i}")
        # print(f"更新类型：{self.type_i}")

        all_possibe_time = [tablea_updae_time, tcol_updae_time, update_time]
        return max(all_possibe_time)






### 压主库性能的负载
# daily 不同步，只压主库性能
# 数据库 dailydata
class DailyWorkLoad:
    def __init__(self, reload):     # 每次运行需要设置为1，不会变动update表。只会先创建4000条数据的delete表，和空的insert表
        # 线程数，      插入,删除, 修改
        self.txntype = [200, 5, 65]   # 标准是 [400, 5, 65]
        self.lock_insert= threading.Lock()
        self.lock_update= threading.Lock()
        self.lock_delete= threading.Lock()
        self.tp_num_insert = 0
        self.tp_num_update = 0
        self.tp_num_delete = 0
        self.insert_sleep = 0
        self.delete_sleep = 0
        self.update_sleep = 0
        if reload == 1:
            self.rebuild_data()

    # insert和delete表清空，再delete表填满
    # "daily"名字开头的表，开始序号的 delete表。接着是insert表
    def rebuild_data(self):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='dailydata', charset='utf8')
        cursor = db.cursor()
        for i in range(self.txntype[1]):  # delete表
            table = 'daily' + str(i)
            sql = "TRUNCATE TABLE %s;" % (table)  # 清空表，速度更快
            cursor.execute(sql)
            db.commit()
            sql = "ALTER TABLE %s AUTO_INCREMENT = 1;" % (table)  # 自增列编码从1开始
            cursor.execute(sql)
            db.commit()
            for r in range(4000):  # 插入数据
                rvalue = np.random.randint(low=10, high=200, size=4) + np.random.rand(4)
                sql = "INSERT INTO %s(h1, h2, h3, h4) VALUES(%s, %s, %s, %s);" % \
                      (table, rvalue[0], rvalue[1], rvalue[2], rvalue[3])
                cursor.execute(sql)
                db.commit()

        for i in range(self.txntype[1], self.txntype[1] + self.txntype[0]):  # insert表
            table = 'daily' + str(i)
            sql = "TRUNCATE TABLE %s;" % (table)  # 清空表，速度更快
            cursor.execute(sql)
            db.commit()
            sql = "ALTER TABLE %s AUTO_INCREMENT = 1;" % (table)  # 自增列编码从1开始
            cursor.execute(sql)
            db.commit()

        cursor.close()
        db.close()

    #插入数量到 10万 就停止,参数表名
    def insert_tp(self, tname):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='dailydata', charset='utf8')
        cursor = db.cursor()
        for i in range(100000):
            time.sleep(self.insert_sleep)
            rvalue = np.random.randint(low=10, high=200, size=4) + np.random.rand(4)
            sql = "INSERT INTO %s(h1, h2, h3, h4) VALUES(%s, %s, %s, %s);" % \
                  (tname, rvalue[0], rvalue[1], rvalue[2], rvalue[3])
            cursor.execute(sql)
            db.commit()
            self.lock_insert.acquire()
            self.tp_num_insert += 1
            self.lock_insert.release()
            time.sleep(0.01)
        cursor.close()
        db.close()

    # 4000条数据随机更新，执行 10万 次停止
    def update_tp(self, tname):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='dailydata', charset='utf8')
        cursor = db.cursor()
        for i in range(100000):
            time.sleep(self.update_sleep)
            rvalue = np.random.randint(low=10, high=200, size=4) + np.random.rand(4)
            rid = random.randint(0, 4000)
            sql = "UPDATE %s SET h1=%s,h2=%s,h3=%s,h4=%s WHERE id=%s;" % \
                  (tname, rvalue[0], rvalue[1], rvalue[2], rvalue[3], rid)
            cursor.execute(sql)
            db.commit()
            self.lock_update.acquire()
            self.tp_num_update += 1
            self.lock_update.release()
            time.sleep(0.01)
        cursor.close()
        db.close()

    # 在4000条数据中随机删除一条，直到剩余100条停止
    def delete_tp(self, tname):
        db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                             database='dailydata', charset='utf8')
        cursor = db.cursor()
        for i in range(3900):
            time.sleep(self.delete_sleep)
            sql = "DELETE FROM %s ORDER BY rand() LIMIT 1;" % (tname)
            cursor.execute(sql)
            db.commit()
            self.lock_delete.acquire()
            self.tp_num_delete += 1
            self.lock_delete.release()
            time.sleep(0.02)
        cursor.close()
        db.close()


    def run(self):
        daily_thread = []
        for i in range(self.txntype[2]):
            table = 'updatetable' + str(i)
            daily_thread.append(threading.Thread(target=self.update_tp, args=(table,), daemon=True))
        for i in range(self.txntype[1]):
            table = 'daily' + str(i)
            daily_thread.append(threading.Thread(target=self.delete_tp, args=(table,), daemon=True))
        for i in range(self.txntype[1], self.txntype[1]+self.txntype[0]):
            table = 'daily' + str(i)
            daily_thread.append(threading.Thread(target=self.insert_tp, args=(table,), daemon=True))

        for t in daily_thread:
            t.start()



# 用来重新创建dailydata库数据。只在主加载数据
def load_dailydata():
    # 清空库里数据
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='dailydata', charset='utf8')
    cursor = db.cursor()
    sql = "SELECT concat('drop table ',table_name,';') FROM information_schema.TABLES WHERE table_schema='dailydata';"
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in result:
        cursor.execute(i[0])
        db.commit()
    cursor.close()
    db.close()

    # 修改表updatetable+编号： 0~txntype【2】-1。  删除表daily+编号： txntype【1】-1 。  剩余是 插入表
    txntype = [400, 5, 65]  # 插入、删除、修改
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123', database='dailydata', charset='utf8')
    cursor = db.cursor()

    for i in range(txntype[2]):
        sql = "CREATE TABLE %s (id int NOT NULL AUTO_INCREMENT PRIMARY KEY, h1 double(10,2), h2 double(10,2), " \
              "h3 double(10,2), h4 double(10,2));" % ('updatetable' + str(i))
        cursor.execute(sql)
        db.commit()
    table_num = txntype[0] + txntype[1]
    for i in range(table_num):
        sql = "CREATE TABLE %s (id int NOT NULL AUTO_INCREMENT PRIMARY KEY, h1 double(10,2), h2 double(10,2), " \
              "h3 double(10,2), h4 double(10,2));" % ('daily' + str(i))
        cursor.execute(sql)
        db.commit()

    for i in range(txntype[2]):
        for r in range(4000):
            rvalue = np.random.randint(low=10, high=200, size=4) + np.random.rand(4)
            sql = "INSERT INTO %s(h1, h2, h3, h4) VALUES(%s, %s, %s, %s);" % \
                  ('updatetable' + str(i), rvalue[0], rvalue[1], rvalue[2], rvalue[3])
            cursor.execute(sql)
            db.commit()
    for i in range(txntype[1]):
        for r in range(4000):
            rvalue = np.random.randint(low=10, high=200, size=4) + np.random.rand(4)
            sql = "INSERT INTO %s(h1, h2, h3, h4) VALUES(%s, %s, %s, %s);" % \
                ('daily' + str(i), rvalue[0], rvalue[1], rvalue[2], rvalue[3])
            cursor.execute(sql)
            db.commit()
    cursor.close()
    db.close()




if __name__ == '__main__':
    # queue_wk = QueueWorkload()
    # queue_wk.create_queuedata([1], 5, '2021-01-01 05:59:50', 10)  # 插入时间是加了间隔载插入
    # queue_wk.create_queuedata([2], 8, '', 20)
    # queue_wk.create_queuedata([1], 4, '', 5)
    # queue_wk.create_queuedata([3], 2, '', 90)
    # queue_wk.create_queuedata([2], 4, '', 30)
    # queue_wk.create_queuedata([3], 8, '', 60)
    # queue_wk.queuing_syn()
    load_dailydata()