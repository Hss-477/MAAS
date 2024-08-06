import ast
import random
from datetime import datetime, timedelta
import pandas as pd


# 仓库表
# warehouseid、warehouse_name、park_count
import pymysql

from Wait.parameter import sql_create_warehouse_info, sql_insert_warehouse_info, sql_create_cargo_info, \
    sql_insert_cargo_info


def create_warehouse():
    df_warehouse = pd.read_excel('/root/dataset_accu/queue/warehouse_info.xlsx',
                                 usecols=[2, 4], names=['warehouse_name', 'park_count'])
    # 去除重复行
    warehouseinfo = df_warehouse.drop_duplicates(subset='warehouse_name')
    # 增加id列，并给列换顺序
    warehouseinfo['warehouseid'] = range(1, len(warehouseinfo)+1)
    warehouseinfo = warehouseinfo.reindex(columns=['warehouseid', 'warehouse_name', 'park_count'])  #换位

    return warehouseinfo
    # 到这里保存文件后，根据info手动补全cargo_kind列，表示这个仓库可以提供的品类，还会处理一些没有的记录的仓库

# 货物表
# cargoid、cargo_name、loading_time
def create_cargo():
    df_single = pd.read_excel('/root/dataset_accu/queue/info_single.xlsx',
                              usecols=[4], names=['cargo_name'])
    cargoinfo = df_single.drop_duplicates(subset='cargo_name')
    cargoinfo['cargoid'] = range(1, len(cargoinfo)+1)
    # 设置货物的装载时间是在10分钟到两个小时之间
    cargoinfo['loading_time'] = [random.randint(10, 120) for _ in range(len(cargoinfo))]
    cargoinfo = cargoinfo.reindex(columns=['cargoid', 'cargo_name', 'loading_time'])
    # print(cargoinfo)

    return cargoinfo

# 排队表（初始化）
# 队列表
# 记录id、用户、货物、仓库
# 、开始排队、入厂时间、入仓时间、离厂时间
# recordid、user_code、cargo_name、warehouse_name
# 、queue_start_time、entry_time、entry_whouse_time、finish_time
# 会初始化一些数据，可指定仓库及排队中到仓库的车辆数
#### 这个函数在运行时暂时不使用了，此部分功能已加入到QueueWorkload中
def create_queuing(warehouse_num, num, sec):
    db = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                         database='accuqueuedata', charset='utf8')
    cursor = db.cursor()

    #初始化，userid是 U000 按顺序生成，
    # 每隔5秒来一个车，库【1,5,10,15,20,25,30,35,40,45,50】下的有15车，物品循环，然后打乱放进去再加算时间
    queue_df = pd.DataFrame(columns=['user_code', 'cargo_name', 'warehouse_name'])
    # warehouse_num = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # warehouse_num = [6, 21]
    # num = 15
    cargo_all = list()   # 总的初始化，包含指定的所有的仓库
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
    for i in range(len(warehouse_num) * num):
        user_all.append(f"U{i:03}")

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
    current_time = datetime.strptime('2021-01-01 05:00:00', '%Y-%m-%d %H:%M:%S')  # 解析起始时间字符串
    for _ in range(len(user_all)):
        start_time_all.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))  # 格式化当前时间并添加到列表中
        current_time += timedelta(seconds=sec)  # 增加指定的时间间隔
    shuffled_df['start_time'] = start_time_all
    # print(shuffled_df)

    # 创建queuing_info表并将上述内容插入
    sql = "DROP TABLE IF EXISTS queuing_info;"
    cursor.execute(sql)
    db.commit()
    sql = "CREATE TABLE queuing_info (recordid int NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
          "user_code varchar(10), cargo_name varchar(255), warehouse_name varchar(255), " \
          "start_time datetime, entry_time datetime, entry_whouse_time datetime, finish_time datetime);"
    cursor.execute(sql)
    db.commit()

    for _, r in shuffled_df.iterrows():
        sql = "INSERT INTO queuing_info (user_code, cargo_name, warehouse_name, start_time) VALUES(%s, %s, %s, %s);"
        cursor.execute(sql, (r['user_code'], r['cargo_name'], r['warehouse_name'], r['start_time']))
    db.commit()


    ###### 生成少量数据，为了测试其他模块功能，后期可注释 ######
    # 这里格外添加保存在excel，是为了生成小一点的数据，手动进行修改再上传，测试实现的功能是否正确
    # 这一段后期不用了，可以注释掉
    sql = "SELECT * FROM queuing_info;"
    queuing_df = pd.read_sql_query(sql, db)  #这个函数直接将查询结果生成dataframe
    queuing_df.to_excel('/root/dataset_accu/queuing_info.xlsx', index=False, encoding='utf-8')
    #############


    cursor.close()
    db.close()


# 创建库 create database accuqueuedata character set utf8;
def warehousedata_to_mysql():
    for ip in ['106.75.233.244', '106.75.244.49']:
    # for ip in ['106.75.233.244']:
        # for ip in ['10.11.6.119']:  # 只有主
        db = pymysql.connect(host=ip, user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        sql = "DROP TABLE IF EXISTS warehouse_info;"
        cursor.execute(sql)
        db.commit()
        cursor.execute(sql_create_warehouse_info)
        db.commit()

        # 将数据插入到MySQL表中
        df = pd.read_excel('/root/dataset_accu/warehouse_info.xlsx')
        for row in df.itertuples():
            values = (row.warehouseid, row.warehouse_name, row.park_count, row.cargo_kind, row.wt)
            cursor.execute(sql_insert_warehouse_info, values)

        # 提交更改并关闭连接
        db.commit()
        cursor.close()
        db.close()

def cargodata_to_mysql():
    for ip in ['106.75.233.244', '106.75.244.49']:
    # for ip in ['106.75.233.244']:
        # for ip in ['10.11.6.119']:  # 只有主
        db = pymysql.connect(host=ip, user='root', password='huangss123',
                             database='accuqueuedata', charset='utf8')
        cursor = db.cursor()
        sql = "DROP TABLE IF EXISTS cargo_info;"
        cursor.execute(sql)
        db.commit()
        cursor.execute(sql_create_cargo_info)
        db.commit()

        # 将数据插入到MySQL表中
        df = pd.read_excel('/root/dataset_accu/cargo_info.xlsx')
        for row in df.itertuples():
            values = (row.cargoid, row.cargo_name, row.loading_time, row.ct)
            cursor.execute(sql_insert_cargo_info, values)

        # 提交更改并关闭连接
        db.commit()
        cursor.close()
        db.close()

#这里是为了测试写的，使用create_queuing()仅生成少量数据，保存excel后修改，测试其他功能的实现
def queuingdata_to_mysql():
    for ip in ['106.75.233.244']:
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

        # 将数据插入到MySQL表中
        df = pd.read_excel('/root/dataset_accu/queuing_info.xlsx')
        for row in df.itertuples():
            sql = "INSERT INTO queuing_info VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
            values = (row.recordid, row.user_code, row.cargo_name, row.warehouse_name,
                      row.start_time, row.entry_time, row.entry_whouse_time, row.finish_time)
            new_vaules = tuple(None if x != x else x for x in values)
            cursor.execute(sql, new_vaules)

        db.commit()
        cursor.close()
        db.close()


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

    ####### 仓库、货物 ######
    # 1.1） 创建 仓库和货物，这里只是筛选和生成基础信息，对生成的进行手动修改是最后使用的版本
    # warehouse_info = create_warehouse()
    # warehouse_info.to_excel('/root/dataset_accu/warehouse_info.xlsx', index=False, encoding='utf-8')
    # cargo_info = create_cargo()
    # cargo_info.to_excel('/root/dataset_accu/cargo_info.xlsx', index=False, encoding='utf-8')
    # 1.2） 将手动修改好的仓库和货物数据加载到mysql
    warehousedata_to_mysql()
    cargodata_to_mysql()

    ####### 排队（初始化） ######
    # create_queuing()
    # queuingdata_to_mysql()   # 加载少量数据进行测试

