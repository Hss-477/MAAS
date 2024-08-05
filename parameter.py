


# 各种粒度都监督
feature_to_version = ['accuqueuedata',
                      'cargo_info', 'warehouse_info', 'queuing_info',
                      'entry_time', 'entry_whouse_time', 'finish_time',
                      'park_count', 'loading_time']

# 需要建立版本表的属性
# 因为这两个不能通过查询原始数据表获得，所以需要在每次更新的原始数据时手动更新版本表
attr_version = ['park_count', 'loading_time']
sql_create_col = "CREATE TABLE vnum_tcol (qn INT, wn INT, cn INT);"
sql_insert_col = "INSERT INTO vnum_tcol(qn, wn, cn) VALUES(0, 0, 0);"
sql_update_col = "update vnum_tcol set qn = %s, wn = %s, cn = %s;"
# primary '106.75.233.244'   backup '106.75.244.49'




sql_create_warehouse_info = "CREATE TABLE warehouse_info (warehouseid tinyint PRIMARY KEY, " \
                            "warehouse_name varchar(255), park_count tinyint, cargo_kind varchar(255), " \
                            "wt datetime);"
sql_insert_warehouse_info = "INSERT INTO warehouse_info  VALUES (%s, %s, %s, %s, %s);"


sql_create_cargo_info = "CREATE TABLE cargo_info (cargoid tinyint PRIMARY KEY, " \
                        "cargo_name varchar(255), loading_time tinyint, " \
                        "ct datetime);"
sql_insert_cargo_info = "INSERT INTO cargo_info VALUES (%s, %s, %s, %s)"


sql_create_queuing_info = "CREATE TABLE queuing_info (recordid int NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
                          "user_code varchar(10), cargo_name varchar(255), warehouse_name varchar(255), " \
                          "start_time datetime, entry_time datetime, entry_whouse_time datetime, finish_time datetime, " \
                          "qt datetime);"
sql_insert_queuing_info = "INSERT INTO queuing_info VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"


sql_create_tablea = "CREATE TABLE tablea (tid int NOT NULL AUTO_INCREMENT PRIMARY KEY, t1 double(10,2), " \
                    "t2 double(10,2), t3 double(10,2), t4 double(10,2), t5 datetime);"
sql_insert_tablea = "INSERT INTO tablea(t1, t2, t3, t4, t5) VALUES (%s, %s, %s, %s, %s);"





# 构建感知服务类需要的信息
# 1. collect4.py收集样本数据文件的列名
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
# 2. 建立归一化处理器和独热编码器的列名基础。除了最后一个’error‘
encoder_colums = [
    'vtime_park_count', 'vtime_loading_time',
    'vtime_start_time', 'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_finish_time',
    'vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info', 'vtime_tablea',
    'vtime_accuqueuedata',
    'vnum_park_count', 'vnum_loading_time',
    'vnum_start_time', 'vnum_entry_time', 'vnum_entry_whouse_time', 'vnum_finish_time',
    'vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info', 'vnum_tablea',
    'vnum_accuqueuedata',
    'Q1_1_time', 'Q1_2_time', 'Q1_3_time', 'Q1_1_num', 'Q1_2_num', 'Q1_3_num',
    'Q2_1_time', 'Q2_2_time', 'Q2_3_time', 'Q2_1_num', 'Q2_2_num', 'Q2_3_num',
    'Q3_1_time', 'Q3_2_time', 'Q3_3_time', 'Q3_1_num', 'Q3_2_num', 'Q3_3_num',
    'now_day', 'now_hour', 'now_minute', 'now_second',
    'start_day', 'start_hour', 'start_minute', 'start_second',
    'warehouse_name',
    'park_count_p', 'park_count_s',
    'status1_p_1', 'status1_p_2',
    'status1_s_1','status1_s_2',
    'status2_p_1','status2_p_2','status2_p_3','status2_p_4','status2_p_5',
    'status2_p_6','status2_p_7','status2_p_8','status2_p_9','status2_p_10',
    'status2_s_1','status2_s_2','status2_s_3','status2_s_4','status2_s_5',
    'status2_s_6','status2_s_7','status2_s_8','status2_s_9','status2_s_10',
    'status3_p', 'status3_s',
    'error']
# 3. 数据处理后的列名（归一化，独热编码）
sample_data_colums_after = [
        'vtime_park_count', 'vtime_loading_time',
        'vtime_start_time', 'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_finish_time',
        'vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info', 'vtime_tablea',
        'vtime_accuqueuedata',
        'vnum_park_count', 'vnum_loading_time',
        'vnum_start_time', 'vnum_entry_time', 'vnum_entry_whouse_time', 'vnum_finish_time',
        'vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info', 'vnum_tablea',
        'vnum_accuqueuedata',
        'Q1_1_time', 'Q1_2_time', 'Q1_3_time', 'Q1_1_num', 'Q1_2_num', 'Q1_3_num',
        'Q2_1_time', 'Q2_2_time', 'Q2_3_time', 'Q2_1_num', 'Q2_2_num', 'Q2_3_num',
        'Q3_1_time', 'Q3_2_time', 'Q3_3_time', 'Q3_1_num', 'Q3_2_num', 'Q3_3_num',
        'now_day', 'now_hour', 'now_minute', 'now_second',
        'start_day', 'start_hour', 'start_minute', 'start_second',
        'warehouse_name',
        'park_count_p', 'park_count_s',
        'status1_p_1', 'status1_p_2', 'status1_p_3', 'status1_p_4', 'status1_p_5',
        'status1_p_6', 'status1_p_7', 'status1_p_8', 'status1_p_9', 'status1_p_10',
        'status1_s_1', 'status1_s_2', 'status1_s_3', 'status1_s_4', 'status1_s_5',
        'status1_s_6', 'status1_s_7', 'status1_s_8', 'status1_s_9', 'status1_s_10',
        'status2_p_1', 'status2_p_2', 'status2_p_3', 'status2_p_4', 'status2_p_5',
        'status2_p_6', 'status2_p_7', 'status2_p_8', 'status2_p_9', 'status2_p_10',
        'status2_s_1', 'status2_s_2', 'status2_s_3', 'status2_s_4', 'status2_s_5',
        'status2_s_6', 'status2_s_7', 'status2_s_8', 'status2_s_9', 'status2_s_10',
        'status3_p', 'status3_s',
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
        'warehouse_name_高线库_一线', 'warehouse_name_高线库_三线', 'warehouse_name_高线库_下线库', 'warehouse_name_高线库_二线',
        'error']    # 注意状态3的主备值，就只有一位，不用拆分，用收集到的即可


# 4.新鲜度特征，不同度量方式的
# 三个级别，Time和Num方式都用，没有分开
feature_vgap_tn = [['vtime_accuqueuedata',
                    'vnum_accuqueuedata'],
                   ['vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info',
                    'vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info'],
                   ['vtime_park_count', 'vtime_loading_time', 'vtime_start_time', 'vtime_entry_time',
                    'vtime_entry_whouse_time', 'vtime_finish_time',
                    'vnum_park_count', 'vnum_loading_time', 'vnum_start_time', 'vnum_entry_time',
                    'vnum_entry_whouse_time', 'vnum_finish_time']]
# 仅使用一个度量
feature_vgap_t = [['vtime_accuqueuedata'],
                  ['vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info'],
                  ['vtime_park_count', 'vtime_loading_time', 'vtime_start_time', 'vtime_entry_time',
                   'vtime_entry_whouse_time', 'vtime_finish_time']]
feature_vgap_n = [['vnum_accuqueuedata'],
                  ['vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info'],
                  ['vnum_park_count', 'vnum_loading_time', 'vnum_start_time', 'vnum_entry_time',
                   'vnum_entry_whouse_time', 'vnum_finish_time']]
# 5. 额外特征
# 一个静态，动态三个，  虽然均有主备两个值，但是目前就当做独立的，这里需要进一步思考代价
# 一些特征需要主要两个值，一些只有一个值
# 按个数，再按主备
# 结果由 aware_service.py文件的feature_neednum函数结果得到
# feature_add = [['now_time', 'start_time', 'warehouse_name'],
#                ['park_count_s'], ['park_count_p'],
#                ['status3_s'], ['status3_p'],
#                ['status2_s'], ['status2_p'],
#                ['status1_p'], ['status1_s']]
feature_add = [['start_time', 'warehouse_name', 'now_time'],
               ['park_count_s'], ['park_count_p'],
               ['status3_s'], ['status3_p'],
               ['status2_s'], ['status2_p'],
               ['status1_p'], ['status1_s']]
# ['park_count_s', 'park_count_p', 'status3_s', 'status3_p', 'status2_s', 'status2_p', 'status1_p', 'status1_s']
#[1.0, 1.0, 1.705336426914153, 1.7616350484509349, 6.89480687866794, 7.179575542513989, 13.786611164187253, 16.111164187252626]
# 第二轮
#['park_count_s', 'park_count_p', 'status3_s', 'status3_p', 'status2_s', 'status2_p', 'status1_p', 'status1_s']
# [1.0, 1.0, 1.5888, 1.631847619047619, 6.944457142857143, 7.251619047619047, 16.033257142857142, 18.20537142857143]



# 记一个vgap的顺序    单个级别下是11位数据
# 'vtime_park_count', 'vtime_loading_time',
# 'vnum_park_count', 'vnum_loading_time',
# 'vtime_start_time', 'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_finish_time',
# 'vnum_start_time', 'vnum_entry_time', 'vnum_entry_whouse_time', 'vnum_finish_time',
# 'vtime_queuing_info', 'vtime_cargo_info', 'vtime_warehouse_info', 'vtime_tablea',
# 'vnum_queuing_info', 'vnum_cargo_info', 'vnum_warehouse_info', 'vnum_tablea',
# 'vtime_accuqueuedata',
# 'vnum_accuqueuedata',

# 记attrextend顺序
# ['vtime_park_count',
#  'vtime_start_time', 'vtime_entry_time', 'vtime_loading_time',
#  'vtime_entry_time', 'vtime_entry_whouse_time', 'vtime_loading_time',
#  'vtime_entry_whouse_time', 'vtime_finish_time', 'vtime_loading_time']







