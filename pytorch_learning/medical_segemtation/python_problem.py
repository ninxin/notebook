"""
1
"""
case1 = [9, 8, 7, 3, 2, 2]
case1 = case1.append(1)  # 没有返回值，None，所以不能赋值
print(case1)  # None

"""
2
"""
case2 = [9, 8, 8, 3, 3, 1]
for i in case2:
    if i % 2 == 0:
       case2.remove(i)      # 删除后，后面的元素会前移

print(case2)        # [9, 8, 3, 3, 1]
# 修改后
case2 = [9, 8, 8, 3, 3, 1]
tmp = [x for x in case2 if x % 2 != 0]
print(tmp)

"""
3
"""
case3 = ['a', 'b' 'c', 'd', 'e']
print(len(case3))       # 4

"""
4
"""
case4 = ('bilibili')        # 字符串
for i in case4:
    print(i)        # 将每个字符逐行打印
# 改为元组后
case4 = tuple('bilibili')       # 变为元组还是不行
for i in case4:
    print(i)

# 想在不可变对象元组中添加一个可变对象时，应该加个,
case4 = ('bilibili',)        # 元组
for i in case4:
    print(i)            # bilibili，就不会逐行输出

"""
5
"""
flag = True

# if flag:
#     x, y = 10, 10
# else:
#     x, y = None, None
x, y = (10, 10) if flag else None, None     # 看不到后一个None，改为(None, None)

print(x, y)     # (10, 10) None

"""
6
"""
data = [1, 4, 5, 7, 9]
# 将奇数变为0
for i in range(len(data)):
    if data[i] % 2:
        data[i] = 0
# 更好的方法,使用内置的枚举函数，直接获得相应值及索引
for idx, num in enumerate(data):
    if num % 2:
        data[idx] = 0
print(data)     # [0, 4, 0, 0, 0]

"""
7
"""
# 对可迭代对象排序，如：列表，元组，字典
# 列表
data = [-1, -10, 0, 9, 5]
new_data = sorted(data)     # 升序，若要降序，将reverse设为true
print(new_data)
# 元组
data = (-1, -10, 0, 9, 5)
new_data = sorted(data)
print(new_data)     # 输出类型变为列表
# 复杂的可迭代对象排序，如列表中的对象为字典类型
data = [
    {"name": "jia", "age": 18},
    {"name": "yi", "age": 80},
    {"name": "bing", "age": 39}
]
new_data = sorted(data, key=lambda x: x["age"])     # 按年纪排序
print(new_data)

"""
8
"""
# 将列表中的重复值去重
data = [1, 1, 2, 3, 4, 5, 5]
# & 交集  | 并集    - 差     ^ 补集
new_data = set(data)        # 集合，没有重复值
print(new_data)

"""
9
"""
# 用sum计算多个数字的列表，可能会出现内存错误
# 将中括号改成小括号，变为生成器
import sys

data = [i for i in range(10000)]
print(sys.getsizeof(data), "bytes")     # 87616,所占的空间
data = (i for i in range(10000))        # 可以提高效率
print(sys.getsizeof(data), "bytes")     # 112

print(sum(data))

"""
10
"""
# 获取字典中不存在的建
data = {"name": "ads", "age": "10"}
# uid = data["uid"]
uid = data.get("uid", "000")        # 若索引不到"uid",则返回 000
uid = data.setdefault("uid", "000")     # 若没有这个建，则添加，并将之设置为默认值，即 000
print(data)

"""
11
"""
# 将列表中的字符串拼接
data = ['hi', 'my', 'data']
newdata = ''
for i in data:
    newdata += i + ' '      # 中间用空格连接
print(newdata)
# 对于大型字符串慢
new_data = " ".join(data)
print(new_data)

"""
12
"""
# 合并两个字典,不用遍历对比判断键
data1 = {"name": "asd", "age": "10"}
data2 = {"name": "doa", "uid": "1891"}
outdata = {**data1, **data2}
print(outdata)

"""
13
"""
# 将某对象与多个对象进行对比判断
data = 'a'
if data == 'a' or data == 'b' or data == 'c':
    print('hah')
# 修改后
datas = ['a', 'b', 'c']
data = 'a'
if data in datas:
    print('hah')

"""
14
"""
# 读取文件
# 这种写入错误时，文件不会被关闭
def case_14(filepath):
    f = open(filepath, "w")
    f.write("hello world\n")
    f.close()

def case_14(filepah):
    with open(filepah) as f:
        f.write("hello world\n")

"""
15
数值大的变量
是用下划线更加简洁明了
"""
x = 10000000
x = 10_000_000

"""
16
定义函数时，若存在可变类型参数要小心
需要对其拷贝
"""
def case_16(n, lst=[]):
    lst.append(n)
    return lst
lst1 = case_16(0)   # [0]
lst2 = case_16(1)   # [0, 1]

def case_16(n, lst=None):
    if lst is None:
        lst = []
    lst.append(n)
    return lst
lst1 = case_16(0)   # [0]
lst2 = case_16(1)   # [1]

"""
17
统计运行的时间
用time.perf_counter()更准确
"""
import time

def case_17():
    t1 = time.time()
    time.sleep(1)
    t2 = time.time()
    print(t2-t1)

def case_17():
    t1 = time.perf_counter()
    time.sleep(1)
    t2 = time.perf_counter()
    print(t2 - t1)


"""
18
用isinstance代替==判断类型，避免一些子类带来的问题
"""
from collections import namedtuple

line = namedtuple('line', ['k', 'b'])
l = line(1, 5)
if type(l) == tuple:
    print('is')
else:
    print('not')    # not

if isinstance(l, tuple):
    print('is')     # is
else:
    print('not')

"""
19
列表推导式
"""
symbole = '^$(&#%'
codes = []

for sym in symbole:
    codes.append(ord(symbole))

symbole = '^$(&#%'
codes = [ord(symbole) for sym in symbole]

"""
20
不可变;
Number 数值
String 字符串
Tuple 元组

可变：
List 列表
Dictionary 字典
"""
