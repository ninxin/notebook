# 占位符
name = 'nick'
age = 19
print('my name is %s my age is %s' % (name, age))
# format格式化
name = 'nick'
age = 19
print("Hello, {}. You are {}.".format(name, age))
# f-String格式化
name = "nick"
age = 19
print(f"Hello, {name}. You are {age}.")

salary = 6.6666
print(f'{salary:.2f}')

# 按位运算符: 与,或,异或,非
a = 60    # 0011 1100
b = 13    # 0000 1101
a&b       # 0000 1100   12
a|b       # 0011 1101   61
a^b       # 0011 0001   49
~a        # 1100 0011

"""
字符串
str.strip('*')     移出*，默认空格
str.split('*', num)     以*为分界点，从左到右分割成num+1份
str.lstrip('*')        从左边开始移出*,没有则停止
str.rstrip('*')        从右
str.lower()            小写
str.upper()            大写
str.startswith('*')     判断是否是*开头
str.endswith('*')       判断是否是*结尾
str.rsplit('*')         从头开始分割
str.join('*')           拼接
str.replace('*')        替代

# True: Unicode数字，byte数字（单字节），全角数字（双字节），罗马数字
# False: 汉字数字
# Error: 无
str.isdigit()      

# True: Unicode数字，全角数字（双字节）
# False: 罗马数字，汉字数字
# Error: byte数字（单字节）     
str.isdecimal()

# True: Unicode数字，全角数字（双字节），罗马数字，汉字数字
# False: 无
# Error: byte数字（单字节）
str.isnumeric()

str.find(‘*’)           找到*位置的索引，找不到返回-1
str.rfind(‘*’)          从右找
str.index(‘*’)          返回位置的索引
str.rindex(‘*’)         从右数的索引
str.count(‘*’)          *出现的次数
str.center(50, '*')     字符串在中间，两边用50个*填充
str.ljust(50, '*')      在左边
str.rjust(50, '*')      在右边
str.zfill()             字符串在右边，左边用0填充
str.captalize()         首字母大写其余小写（一串字符串只有一个）
str.swapcase()          大小写互换
str.title()             首字母大写其余小写（可能有多个大写的，比如空格后的字符）
str.isalnum()           至少有一个字符并且所有字符都是字母或数字则返回True
str.isalpha()           至少有一个字符并且所有字符都是字母则返回True
"""
# 可变: 字典，列表， 集合     不可变:数字， 字符串， 元组
# 浅拷贝:如果l2是l1的浅拷贝对象，则l1内的不可变元素发生了改变，l2不变；如果l1内的可变元素发生了改变，则l2会跟着改变
# 深拷贝不会变
import copy
# 列表可变
l1 = [1, 'a', (1,), [1, 2], {1, 2}]
l2 = l1
l3 = copy.copy(l1)
l4 = copy.deepcopy(l1)
print(f'id(l1):{id(l1)}, id(l2):{id(l2)}, id(l3):{id(l3)}, id(l4):{id(l3)}')     # l2相同， l3不同, l4不同
l1[0] = 3
l1[1] = 'c'
l1[2] = (3,)
l1[3] = [4, 5]
l1[4] = {4, 5}
print(f'l1:{l1}, \nl2:{l2}, \nl3:{l3}, \nl4:{l4}')          # l2变, l3不变, l4不变
l2[0] = -1
l3[1] = 'z'
print(f'l1:{l1}')
print(f'id(l1):{id(l1)}, id(l2):{id(l2)}, id(l3):{id(l3)}, id(l4):{id(l4)}')        # 深拷贝浅拷贝地址都不同
# 字符串不可变
l1 = 'a'
l2 = copy.copy(l1)
l3 = copy.deepcopy(l1)
print(f'id(l1):{id(l1)}, id(l2):{id(l2)}, id(l3):{id(l3)}')     # 地址都相同

chr(65)         # 参考ASCII码表将数字转成对应字符
ord('A')        # 将字符转换成对应的数字

eval('[1,2,3]')     # 把字符串翻译成数据类型
hash()              # 是否可哈希
all()               # 可迭代对象内元素全为真，则返回真
any()               # 可迭代对象中有一元素为真，则为真
bin()               # 二进制转换
oct()               # 八进制转换
hex()               # 十六进制转换
dir()               # 包内的函数
frozenset()         # 不可变集合





# 冒泡排序
def bubble_sort(list):
    unsorted_until_index = len(list) - 1
    sorted = False
    while not sorted:
        sorted = True
        for i in range(unsorted_until_index):
            if list[i] > list[i+1]:
                sorted = False
                list[i], list[i+1] = list[i+1], list[i]
        unsorted_until_index = unsorted_until_index - 1
list = [65, 55, 45, 35, 25, 15, 10]
bubble_sort(list)
print(list)

# 插入排序
def insertion_sort(array):
    for index in range(1, len(array)):
        position = index
        temp_value = array[index]
        while position > 0 and array[position - 1] > temp_value:
            array[position] = array[position - 1]
            position = position - 1
        array[position] = temp_value
