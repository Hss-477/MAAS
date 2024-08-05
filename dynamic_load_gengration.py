import random
import ast
from datetime import datetime


##### 这个函数不使用，已加入到workload.py的class QueueWorkload2中


# 动态生成 queuing环境
# 参数1：将一天的时间划分为车辆到来速度的快慢。为1会生成新文件，再读入作备用。否则直接读取
#      包含： 时间点三个file，快、慢、很慢
#            新车插入三个file，插入库量，插入车数、插入类型（有影响、无影响、混合）
from Wait.workload import QueueWorkload


class DynLoadCreate:
    def __init__(self, basedata, now_time):
        self.basedata = basedata
        self.now_time = now_time
        self.freq_fast = list()      # 每条数据更新时间
        self.freq_slow = list()
        self.freq_veryslow = list()
        self.fast_i = 0    # 时间点获取下标
        self.slow_i = 0
        self.veryslow_i = 0

        self.change_num_wh = list()   # 每次gap插入的新库数量
        self.change_num_car = list()   # 每次gap插入的新车数量
        self.change_type = list()  # 插入类型：有影响、无影响、混合
        self.wh_i = 0
        self.car_i = 0
        self.type_i = 0

        self.create_freqfile()
        self.create_changefile()

    def create_freqfile(self):
        # 创建基础的更改频率时间文件
        if self.basedata == 1:
            freq_fast = [random.randint(10, 180) for _ in range(20)]  # 8万
            print(f"freq_fast: {freq_fast}")
            freq_slow = [random.randint(300, 480) for _ in range(20)]  # 4万
            print(f"freq_slow: {freq_slow}")
            freq_veryslow = [random.randint(600, 1200) for _ in range(20)]  # 2千
            print(f"freq_veryslow: {freq_veryslow}")
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

    def create_changefile(self):
        # 操作同上，只是file用作不用，分开写好区分
        if self.basedata == 1:
            change_num_wh = [random.randint(1, 4) for _ in range(20)]  # 5千
            print(f"change_num_wh: {change_num_wh}")
            change_num_car = [random.randint(2, 8) for _ in range(20)]  # 10万    下限不能是1，否则gap无效
            print(f"change_num_car: {change_num_car}")
            change_type = [random.randint(1, 3) for _ in range(5)]  # 2万
            print(f"change_type: {change_type}")
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


    def run(self):
        t6 = datetime.strptime('2021-01-01 06:00:00', '%Y-%m-%d %H:%M:%S')
        t9_30 = datetime.strptime('2021-01-01 09:30:00', '%Y-%m-%d %H:%M:%S')
        t11 = datetime.strptime('2021-01-01 11:00:00', '%Y-%m-%d %H:%M:%S')
        t14 = datetime.strptime('2021-01-01 14:00:00', '%Y-%m-%d %H:%M:%S')
        t16 = datetime.strptime('2021-01-01 16:00:00', '%Y-%m-%d %H:%M:%S')
        t18 = datetime.strptime('2021-01-01 18:00:00', '%Y-%m-%d %H:%M:%S')
        t20_30 = datetime.strptime('2021-01-01 20:30:00', '%Y-%m-%d %H:%M:%S')
        t21_30 = datetime.strptime('2021-01-01 21:30:00', '%Y-%m-%d %H:%M:%S')

        flag = ''
        if (self.now_time > t6 and self.now_time < t9_30) \
                or (self.now_time > t14 and self.now_time < t16) \
                or (self.now_time > t18 and self.now_time < t20_30):
            print("fast")
            flag = 'f'
        elif (self.now_time > t9_30 and self.now_time < t11) \
                or (self.now_time > t16 and self.now_time < t18) \
                or (self.now_time > t20_30 and self.now_time < t21_30):
            print("slow")
            flag = 's'
        elif self.now_time > t11 and self.now_time < t14:
            print("veryslow")
            flag = 'v'

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
            has_wh = self.change_num_car[self.car_i: self.change_num_wh[self.wh_i]]
            has_time = freq_list[freq_i: self.change_num_wh[self.wh_i]]
            self.now_time = update_newtrucks(self.now_time, has_wh, has_time, [])
        elif self.change_type[self.type_i] == 2:
            no_wh = freq_list[freq_i: self.change_num_wh[self.wh_i]]
            self.now_time = update_newtrucks(self.now_time, [], [], no_wh)
        else:
            has_wh = self.change_num_car[self.car_i: self.change_num_wh[self.wh_i]]
            has_time = freq_list[freq_i: self.change_num_wh[self.wh_i]]
            # 因为下标需要再次使用，所以这里也要更新
            if flag == 'f':
                self.fast_i += self.change_num_wh[self.wh_i]
            elif flag == 's':
                self.fast_i += self.change_num_wh[self.wh_i]
            else:
                self.fast_i += self.change_num_wh[self.wh_i]
            freq_i += self.change_num_wh[self.wh_i]
            self.wh_i += 1
            no_wh = freq_list[freq_i: self.change_num_wh[self.wh_i]]
            self.now_time = update_newtrucks(self.now_time, has_wh, has_time, no_wh)

        # 下次使用
        self.type_i += 1
        self.car_i += self.change_num_wh[self.wh_i]
        if flag == 'f':
            self.fast_i += self.change_num_wh[self.wh_i]
        elif flag == 's':
            self.fast_i += self.change_num_wh[self.wh_i]
        else:
            self.fast_i += self.change_num_wh[self.wh_i]
        self.wh_i += 1

        #更新主库并收集数据

        #同步备库，进入下轮收集

        print(self.fast_i)
        print(self.slow_i)
        print(self.veryslow_i)
        print(self.wh_i)
        print(self.car_i)
        print(self.type_i)
        print(self.now_time)


if __name__ == "__main__":
    queue_wk = QueueWorkload()  # 一个空queuing_info表
