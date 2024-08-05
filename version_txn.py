import threading
import time
import pymysql
import warnings
from pymysqlreplication import BinLogStreamReader
from pymysqlreplication.row_event import DeleteRowsEvent, UpdateRowsEvent, WriteRowsEvent

from Wait.parameter import attr_version, feature_to_version, sql_create_col, sql_insert_col, sql_update_col

warnings.filterwarnings('ignore')


# 监听事务维护版本信息，不指定方式和级别，所有的都收取
# Num、Time
# db、rel、attr
# 不用实时监听，直接查表就可以观察到版本信息
# 因为监控中使用的时间戳是事务的时间错，但是需要的是模拟时应该更新的时间
def get_vgap():
    pdb = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                database='accuqueuedata', charset='utf8')
    sdb = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                  database='accuqueuedata', charset='utf8')
    cursor_p = pdb.cursor()
    cursor_s = sdb.cursor()

    # park_count  loading_time这两个属性会创建版本表，每次对他，噩梦进行更新的时候会有对应的sql语句进行版本信息更新
    # 其他的版本信息通过查询可以获得
    # 版本表有：
    #库： accuqueuedata
    #表： queuing_info  cargo_info  warehouse_info  *tablea
    #属性： start_time  entry_time  entry_whouse_time  finish_time  park_count  loading_time
    #      * qt wt ct qn wn cn

    vtime_p = {}
    vnum_p = {}
    vtime_s = {}
    vnum_s ={}
    # 1. 属性 park_count  loading_time
    for f in attr_version:
        name = 'vtime_' + f
        sql = "select * from %s;" % ('accuqueueversion.vtime_' + f)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        vtime_p.update({name: cursor_p.fetchone()[0]})
        vtime_s.update({name: cursor_s.fetchone()[0]})

        name = 'vnum_' + f
        sql = "select * from %s;" % ('accuqueueversion.vnum_' + f)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        vnum_p.update({name: cursor_p.fetchone()[0]})
        vnum_s.update({name: cursor_s.fetchone()[0]})
    # print(vtime_p)
    # print(vtime_s)
    # print(vnum_p)
    # print(vnum_s)
    # 1. 属性：start_time  entry_time  entry_whouse_time  finish_time
    sql = "SELECT MAX(start_time), MAX(entry_time), MAX(entry_whouse_time), MAX(finish_time), MAX(qt) FROM queuing_info;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp = cursor_p.fetchone()
    rs = cursor_s.fetchone()
    vtime_p.update({'vtime_start_time': rp[0], 'vtime_entry_time': rp[1],
                    'vtime_entry_whouse_time': rp[2], 'vtime_finish_time': rp[3]})
    vtime_s.update({'vtime_start_time': rs[0], 'vtime_entry_time': rs[1],
                    'vtime_entry_whouse_time': rs[2], 'vtime_finish_time': rs[3]})
    sql = "SELECT COUNT(start_time) FROM queuing_info WHERE start_time IS NOT NULL;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_start = cursor_p.fetchone()[0]
    rs_start = cursor_s.fetchone()[0]
    sql = "SELECT COUNT(entry_time) FROM queuing_info WHERE entry_time IS NOT NULL;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_entry = cursor_p.fetchone()[0]
    rs_entry = cursor_s.fetchone()[0]
    sql = "SELECT COUNT(entry_whouse_time) FROM queuing_info WHERE entry_whouse_time IS NOT NULL;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_wh = cursor_p.fetchone()[0]
    rs_wh = cursor_s.fetchone()[0]
    sql = "SELECT COUNT(finish_time) FROM queuing_info WHERE finish_time IS NOT NULL;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_finish = cursor_p.fetchone()[0]
    rs_finish = cursor_s.fetchone()[0]
    vnum_p.update({'vnum_start_time': rp_start, 'vnum_entry_time': rp_entry,
                    'vnum_entry_whouse_time': rp_wh, 'vnum_finish_time': rp_finish})
    vnum_s.update({'vnum_start_time': rs_start, 'vnum_entry_time': rs_entry,
                   'vnum_entry_whouse_time': rs_wh, 'vnum_finish_time': rs_finish})
    # print(vtime_p)
    # print(vtime_s)
    # print(vnum_p)
    # print(vnum_s)
    # 2. 表： queuing_info  cargo_info  warehouse_info.
    # vtime 是属性时间，并从中选最大的  （ 注意，对级别的度量，这里是用的是 数据库最新提交事务的时间，而不是属性级别中最旧的时间）
    # vnum 查询列有值个数，求和
    selected_keys_time = ['vtime_start_time', 'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_finish_time']
    selected_keys_num = ['vnum_start_time', 'vnum_entry_time', 'vnum_entry_whouse_time', 'vnum_finish_time']
    maxp = max(vtime_p[key] for key in selected_keys_time if vtime_p[key] is not None)
    maxs = max(vtime_s[key] for key in selected_keys_time if vtime_s[key] is not None)
    sump = sum(vnum_p[key] for key in selected_keys_num)
    sums = sum(vnum_s[key] for key in selected_keys_num)
    # 还要和新增的qt列取得最大的作为表的time度量值
    qt_p = rp[4]
    qt_s = rs[4]
    if qt_p and qt_p >= maxp:
        vtime_p.update({'vtime_queuing_info': qt_p})
    else:
        vtime_p.update({'vtime_queuing_info': maxp})
    if qt_s and qt_s >= maxs:
        vtime_s.update({'vtime_queuing_info': qt_s})
    else:
        vtime_s.update({'vtime_queuing_info': maxs})
    # 还需要把新增的项的修改次数加上
    sql = "select qn, cn, wn from accuqueueversion.vnum_tcol;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_tcol = cursor_p.fetchone()
    rs_tcol = cursor_s.fetchone()
    vnum_p.update({'vnum_queuing_info': sump + rp_tcol[0]})
    vnum_s.update({'vnum_queuing_info': sums + rs_tcol[0]})

    sql = "select MAX(ct) from cargo_info;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_ct = cursor_p.fetchone()[0]
    rs_ct = cursor_s.fetchone()[0]
    p_time_loading = vtime_p.get('vtime_loading_time')
    s_time_loading = vtime_s.get('vtime_loading_time')
    if rp_ct >= p_time_loading:
        vtime_p.update({'vtime_cargo_info': rp_ct})
    else:
        vtime_p.update({'vtime_cargo_info': p_time_loading})
    if rs_ct >= s_time_loading:
        vtime_s.update({'vtime_cargo_info': rs_ct})
    else:
        vtime_s.update({'vtime_cargo_info': s_time_loading})
    vnum_p.update({'vnum_cargo_info': vnum_p.get('vnum_loading_time') + rp_tcol[1]})
    vnum_s.update({'vnum_cargo_info': vnum_s.get('vnum_loading_time') + rs_tcol[1]})

    sql = "select MAX(wt) from warehouse_info;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_wt = cursor_p.fetchone()[0]
    rs_wt = cursor_s.fetchone()[0]
    p_time_park = vtime_p.get('vtime_park_count')
    s_time_park = vtime_s.get('vtime_park_count')
    if rp_wt >= p_time_park:
        vtime_p.update({'vtime_warehouse_info': rp_wt})
    else:
        vtime_p.update({'vtime_warehouse_info': p_time_park})
    if rs_wt >= s_time_park:
        vtime_s.update({'vtime_warehouse_info': rs_wt})
    else:
        vtime_s.update({'vtime_warehouse_info': s_time_park})
    vnum_p.update({'vnum_warehouse_info': vnum_p.get('vnum_park_count') + rp_tcol[2]})
    vnum_s.update({'vnum_warehouse_info': vnum_s.get('vnum_park_count') + rs_tcol[2]})

    # 这个新增表是按时间顺序在增加行数据的，所哟t5存的时间就是表的最新更新时间，tid编号就是更新的条数
    sql = "select t5, tid from tablea ORDER BY tid DESC LIMIT 1;"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    rp_tablea = cursor_p.fetchone()
    rs_tablea = cursor_s.fetchone()
    vtime_p.update({'vtime_tablea': rp_tablea[0]})
    vtime_s.update({'vtime_tablea': rs_tablea[0]})
    vnum_p.update({'vnum_tablea': rp_tablea[1]})
    vnum_s.update({'vnum_tablea': rs_tablea[1]})

    # 3. 库： accuqueuedata
    selected_keys_time = ['vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info', 'vtime_tablea']
    selected_keys_num = ['vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info', 'vnum_tablea']
    maxp = max(vtime_p[key] for key in selected_keys_time)
    maxs = max(vtime_s[key] for key in selected_keys_time)
    sump = sum(vnum_p[key] for key in selected_keys_num)
    sums = sum(vnum_s[key] for key in selected_keys_num)
    vtime_p.update({'vtime_accuqueuedata': maxp})
    vtime_s.update({'vtime_accuqueuedata': maxs})
    vnum_p.update({'vnum_accuqueuedata': sump})
    vnum_s.update({'vnum_accuqueuedata': sums})

    # 计算vgap
    # print("primary新鲜度信息：")
    # print(f"Time:{vtime_p}")
    # print(f"Num:{vtime_p}")
    # print("************")
    # print("secondary新鲜度信息：")
    # print(f"Time:{vtime_s}")
    # print(f"Num:{vtime_s}")
    # print("************")

    vgap_time = {}
    vgap_num = {}
    # 求主备之间的time度量的gap，有值则换为分钟单位，无值则为0
    for key in vtime_p.keys():
        if vtime_p[key] is not None and vtime_s[key] is not None:
            vgap_time[key] = (vtime_p[key] - vtime_s[key]).total_seconds() / 60
        else:
            vgap_time[key] = 0
    # 求主备之间的num度量的gap
    for key in vnum_p:
        if key in vnum_s:
            vgap_num[key] = vnum_p[key] - vnum_s[key]
    # print("vgap：")
    # print(f"Time:{vgap_time}")
    # print(f"Num:{vgap_num}")
    # print("************")

    cursor_p.close()
    cursor_s.close()
    pdb.close()
    sdb.close()

    return vgap_time, vgap_num     # 返回主备版本值的差异

# 配合不使用日志监控的方法，这里只建立cargo_info  warehouse_info属性的版本表
# 这你传入字符串格式的时间是初始化货物表和仓库表版本信息初始化的，也就是构建时的时间
def create_versiontable():
    pdb = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                database='accuqueueversion', charset='utf8')
    sdb = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                  database='accuqueueversion', charset='utf8')
    cursor_p = pdb.cursor()
    cursor_s = sdb.cursor()
    # 删除版本数据库下的所有表
    sql = "SELECT concat('drop table ',table_name,';') FROM information_schema.TABLES " \
          "WHERE table_schema='accuqueueversion';"
    cursor_p.execute(sql)
    cursor_s.execute(sql)
    result_p = cursor_p.fetchall()
    result_s = cursor_s.fetchall()
    for i in result_p:
        cursor_p.execute(i[0])
        pdb.commit()
    for i in result_s:
        cursor_s.execute(i[0])
        sdb.commit()

    for name in attr_version:
        # time
        sql = "CREATE TABLE %s (version DATETIME);" % ("vtime_" + name)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        # sql = "INSERT INTO %s(version) VALUES(%s);" % ("vtime_" + name, time_str)
        sql = "INSERT INTO %s(version) VALUES('2024-01-01 05:59:50');" % ("vtime_" + name)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        # num、order。（order目前先注释）
        sql = "CREATE TABLE %s (version INT);" % ("vnum_" + name)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        # sql = "CREATE TABLE %s (version INT);" % ("vorder_" + name)
        # cursor.execute(sql)
        sql = "INSERT INTO %s(version) VALUES(0);" % ("vnum_" + name)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        # sql = "INSERT INTO %s(version) VALUES(0);" % ("vorder_" + name)
        # cursor.execute(sql)
    pdb.commit()
    sdb.commit()

    # 无关数据项的表的修改次数。建立表vnum_tcol，有三列qn、wn、cn
    cursor_p.execute(sql_create_col)
    cursor_s.execute(sql_create_col)
    cursor_p.execute(sql_insert_col)
    cursor_s.execute(sql_insert_col)
    pdb.commit()
    sdb.commit()

    cursor_p.close()
    cursor_s.close()
    pdb.close()
    sdb.close()

def version_syn():
    primaryDB = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                database='accuqueueversion', charset='utf8')
    secondaryDB = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                  database='accuqueueversion', charset='utf8')
    cursor_P = primaryDB.cursor()
    cursor_S = secondaryDB.cursor()

    version = list()
    for f in attr_version:
        sql = "select * from %s;" % ('vnum_' + f)
        cursor_P.execute(sql)
        sql = "UPDATE %s SET version=%s;" % ('vnum_' + f, cursor_P.fetchone()[0])
        cursor_S.execute(sql)
        secondaryDB.commit()

        # sql = "select * from %s;" % ('vorder_' + f)
        # cursor_P.execute(sql)
        # sql = "UPDATE %s SET version=%s;" % ('vorder_' + f, cursor_P.fetchone()[0])
        # cursor_S.execute(sql)
        # secondaryDB.commit()

        sql = "select * from %s;" % ('vtime_' + f)
        cursor_P.execute(sql)
        t = cursor_P.fetchone()[0].strftime('%Y-%m-%d %H:%M:%S')
        sql = "UPDATE %s SET version=str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s');" % ("vtime_" + f, t)
        cursor_S.execute(sql)
        secondaryDB.commit()

    sql = "select qn, wn, cn from vnum_tcol;"
    cursor_P.execute(sql)
    r = cursor_P.fetchone()
    cursor_S.execute(sql_update_col, (r[0], r[1], r[2]))
    secondaryDB.commit()

    cursor_P.close()
    cursor_S.close()
    primaryDB.close()
    secondaryDB.close()

def show_vresiondata():
    pdb = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                database='accuqueueversion', charset='utf8')
    sdb = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                  database='accuqueueversion', charset='utf8')
    cursor_p = pdb.cursor()
    cursor_s = sdb.cursor()

    for f in attr_version:
        sql = "select * from %s;" % ('vnum_' + f)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        vnum_p = cursor_p.fetchone()[0]
        vnum_s = cursor_s.fetchone()[0]

        sql = "select * from %s;" % ('vtime_' + f)
        cursor_p.execute(sql)
        cursor_s.execute(sql)
        vtime_p = cursor_p.fetchone()[0]
        vtime_s = cursor_s.fetchone()[0]

        print(f"主库上{f}的版本信息：")
        print(f"mun：{vnum_p}， time：{vtime_p}")
        print(f"备库上{f}的版本信息：")
        print(f"mun：{vnum_s}， time：{vtime_s}")
        print("***********")

    cursor_p.close()
    cursor_s.close()
    pdb.close()
    sdb.close()


############### 下以是原版的，实时日志监听的。这里不用看，因为排队场景


# 原始使用的建立完整的版本表
def create_versiontable_0(ip):
    db = pymysql.connect(host=ip, user='root', password='huangss123',
                         database='accuqueueversion', charset='utf8')
    cursor = db.cursor()
    # 删除版本数据库下的所有表
    sql = "SELECT concat('drop table ',table_name,';') FROM information_schema.TABLES " \
          "WHERE table_schema='accuqueueversion';"
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in result:
        cursor.execute(i[0])
        db.commit()

    for name in feature_to_version:
        # time
        sql = "CREATE TABLE %s (version DATETIME);" % ("vtime_" + name)
        cursor.execute(sql)
        sql = "INSERT INTO %s(version) VALUES(now());" % ("vtime_" + name)
        cursor.execute(sql)
        # num、order。（order目前先注释）
        sql = "CREATE TABLE %s (version INT);" % ("vnum_" + name)
        cursor.execute(sql)
        # sql = "CREATE TABLE %s (version INT);" % ("vorder_" + name)
        # cursor.execute(sql)
        sql = "INSERT INTO %s(version) VALUES(0);" % ("vnum_" + name)
        cursor.execute(sql)
        # sql = "INSERT INTO %s(version) VALUES(0);" % ("vorder_" + name)
        # cursor.execute(sql)
    db.commit()

    cursor.close()
    db.close()

# 直接从主库查询versiontable结果保存过去
def version_syn_0():
    primaryDB = pymysql.connect(host='106.75.233.244', user='root', password='huangss123',
                                database='accuqueueversion', charset='utf8')
    secondaryDB = pymysql.connect(host='106.75.244.49', user='root', password='huangss123',
                                  database='accuqueueversion', charset='utf8')
    cursor_P = primaryDB.cursor()
    cursor_S = secondaryDB.cursor()

    version = list()
    for f in feature_to_version:
        sql = "select * from %s;" % ('vnum_' + f)
        cursor_P.execute(sql)
        sql = "UPDATE %s SET version=%s;" % ('vnum_' + f, cursor_P.fetchone()[0])
        cursor_S.execute(sql)
        secondaryDB.commit()

        # version.append(cursor_P.fetchone()[0])
        # sql = "select * from %s;" % ('vorder_' + f)
        # cursor_P.execute(sql)
        # sql = "UPDATE %s SET version=%s;" % ('vorder_' + f, cursor_P.fetchone()[0])
        # cursor_S.execute(sql)
        # secondaryDB.commit()

        version.append(cursor_P.fetchone()[0])
        sql = "select * from %s;" % ('vtime_' + f)
        cursor_P.execute(sql)
        t = cursor_P.fetchone()[0].strftime('%Y-%m-%d %H:%M:%S')
        sql = "UPDATE %s SET version=str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s');" % ("vtime_" + f, t)
        cursor_S.execute(sql)
        secondaryDB.commit()

    cursor_P.close()
    cursor_S.close()
    primaryDB.close()
    secondaryDB.close()


# 从监听的log中获取数据库名、表名、列名，影响行数，更新所有版本表
# 注意，这里的时间戳不能是事务的时间错，因为这个是现在当下的，需要的是模拟时应该更新的时间
# （目前“影响行数”的解析用于对属性的更新，并未写关于Num需要行数的更新。因为实际中order和num一样）
# 监听使用前注意mysql配置文件是否设置了会对该库的操作生成日志
# 这是原始的事务监控和版本信息更新。时间是事务的时间，但是在排队长下，时间是更新的时间，所以直接查表获取，使用get_version函数
def version_maintain_0(serverid, host):
    #连接数据库获取游标
    mysql_connect = {
        "host": host,
        "port": 3306,
        "user": "root",
        "passwd": "huangss123"
    }
    # ignored_schemas = ['versiondata', 'dailydata']
    stream = BinLogStreamReader(
        connection_settings=mysql_connect,
        server_id=serverid,  # slave标识，唯一
        blocking=True,  # 阻塞等待后续事件
        resume_stream=True,  # True为从最新位置读取, 默认False
        only_schemas = ['accuqueuedata'],
        # 设定只监控写操作：增、删、改
        only_events=[
            DeleteRowsEvent,
            UpdateRowsEvent,
            WriteRowsEvent
        ],
    )
    db = pymysql.connect(host=host, user='root', password='huangss123',
                         database='accuqueueversion', charset='utf8')
    cursor = db.cursor()
    # 一条sql语句就是一个event，有自己的信息（库、表、时间），即使多条sql放在有begin和commit里，也是event数等于sql数
    for event in stream:
        # print(event.dump())  # 打印所有信息
        database = event.schema
        table = event.table
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
        attr_list = list()
        affect_rownum = 0   # NUm度量方式是涉及到的元组数。（在负载目前都是一个txn一个元祖）
        sql = " "

        # 一条sql影响了多少行，对每一行进行信息处理
        for row in event.rows:
            # print(row)
            # 首次进入计算各类操作影响的attr_list，之后就是为了统计Num度量所需的事务影响的行数
            if affect_rownum == 0:
                # update事件，得到更新的列名
                if isinstance(event, UpdateRowsEvent):
                    attr_before = row.get('before_values')
                    attr_after = row.get('after_values')
                    before_k = list(attr_before.keys())
                    before_v = list(attr_before.values())
                    after_v = list(attr_after.values())
                    diff_k = list()
                    # 得到update的attr，也就是v不一样的项的k.
                    for i in range(len(before_k)):
                        if before_v[i] != after_v[i]:
                            diff_k.append(before_k[i])
                    attr_list.extend(diff_k)
                # insert事件，得到插入的列名
                elif isinstance(event, WriteRowsEvent):
                    attr_after = row.get('values')
                    k = list(attr_after.keys())
                    v = list(attr_after.values())
                    # 选出insert的属性项，此时就是非None的值
                    insert_kv = [i for i in range(len(v)) if v[i] is not None]
                    for i in range(len(insert_kv)):
                        attr_list.append(k[insert_kv[i]])
                # delete事件，得到表的所有列名
                # 此操作实际没有，所以没有修改
                elif isinstance(event, DeleteRowsEvent):
                    attr_after = row.get('values')
                    k = list(attr_after.keys())
                    attr_list.extend(k)
                affect_rownum += 1
            else:
                affect_rownum += 1


        # 更新版本表
        # 库
        sql = "UPDATE %s SET version=version+%s;" % ("vnum" + database, affect_rownum)
        cursor.execute(sql)
        # sql = "UPDATE %s SET version=version+1;" % ("vorder" + database)
        sql = "UPDATE %s SET version=str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s');" \
              % ("vtime" + database, ts_str)
        cursor.execute(sql)
        # 表
        sql = "UPDATE %s SET version=version+%s;" % ("vnum" + table, affect_rownum)
        cursor.execute(sql)
        # sql = "UPDATE %s SET version=version+1;" % ("vorder" + table)
        sql = "UPDATE %s SET version= str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s');" \
              % ("vtime" + table, ts_str)
        cursor.execute(sql)
        # 属性
        for attr in attr_list:
            if attr in feature_to_version:
                sql = "UPDATE %s SET version=version+%s;" % ("vnum" + attr, affect_rownum)
                cursor.execute(sql)
                # sql = "UPDATE %s SET version=version+1;" % ("vorder" + attr)
                sql = "UPDATE %s SET version= str_to_date(\'%s\','%%Y-%%m-%%d %%H:%%i:%%s');" \
                      % ("vtime" + attr, ts_str)
                cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()

# 为了测试更方便，查询所有版本表的信息
def show_vresiondata_0(ip):
    db = pymysql.connect(host=ip, user='root', password='huangss123',
                                database='accuqueueversion', charset='utf8')
    cursor = db.cursor()

    for f in feature_to_version:

        sql = "select * from %s;" % ('vnum_' + f)
        cursor.execute(sql)
        vnum = cursor.fetchone()[0]

        # sql = "select * from %s;" % ('vorder' + f)
        # cursor.execute(sql)
        # vorder = cursor.fetchone()[0]

        sql = "select * from %s;" % ('vtime_' + f)
        cursor.execute(sql)
        vtime = cursor.fetchone()[0].strftime('%Y-%m-%d %H:%M:%S')

        print(f"{f}的版本信息：")
        print(f"mun：{vnum}， time：{vtime}")

    cursor.close()
    db.close()


if __name__ == "__main__":
    # print("监听开启")
    # thread_vgap_maintain_P = threading.Thread(
    #     target=version_maintain, args=(99, '106.75.233.244'), daemon=True)
    # thread_vgap_maintain_P.start()
    # time.sleep(3000)

    # create_versiontable()
    # show_vresiondata()
    vgap_time, vgap_num = get_vgap()
    print(list(vgap_num.values()))
    print(list(vgap_time.values()))