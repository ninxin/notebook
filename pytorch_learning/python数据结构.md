**异序词检测**

长度相同的两个单词所组成的字母是否相同

```python
# 1. 清点法
# 先将第二个字符串转换为列表（字符串不可修改），然后遍历字符串1的所有字母，如果在列表中，则将列表中的那个字母替换掉，使他不会重复对比。否则说明不是异序词

# 2. 排序法
# 将他们变为列表，再排序，如果一样则是
```



**列表**

|        操作        | O效率 |
| :----------------: | :---: |
|        索引        |   1   |
|      索引赋值      |   1   |
|        追加        |   1   |
|       pop()        |   1   |
|       pop(i)       |   n   |
|  insert(i, item)   |   n   |
|        删除        |   n   |
|        遍历        |   n   |
| 包含(判断是否在内) |   n   |
|        切片        |   k   |
|      删除切片      |   n   |
|      设置切片      |  n+k  |
|        翻转        |   n   |
|        连接        |   k   |
|        排序        | nlogn |
|        乘法        |  nk   |



**字典**

通过键访问元素

| 操作 | O效率 |
| :--: | :---: |
| 复制 |   n   |
| 取值 |   1   |
| 赋值 |   1   |
| 删除 |   1   |
| 包含 |   1   |
| 遍历 |   n   |



**栈**

```python
# 实现栈
# 数组头作为栈的底部，尾作为栈的头部
class Stack:
    def __init__(self):
        self.items = []
    
    # 判断是否为空
    def isEmpty(self):
        return self.items == []
    
    # 入栈
    def push(self, item):
        self.items.append(item)
        
    # 出栈
    def pop(self):
        return self.items.pop()
    # 栈的第一个元素
    def peek(self):
        return self.items[len(self.items)-1]
    
    def size(self):
        return len(self.items)
```

**队列**

```python
# 数组的头作为队列的头，尾作为队列的尾
# 从头部插入，尾部删除
class Queue:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
    
    # 入队
    def enqueue(self, item):
        self.items.insert(0, item)
        
    # 出队
    def dequeue(self):
        return self.items.pop()
    
    def size(self):
        return len(self.items)

```

**双端队列**

```python
class Deque:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
    
    def addFront(self, item):
        self.items.append(item)
        
    def addRear(self, item):
        self.items.insert(0, item)
        
    def removeFront(self):
        return self.items.pop()
    
    def removeRear(self):
        return self.items.pop(0)
    
    def size(self):
        return len(self.items)
```

**链表**

```python
# 节点
# 包含列表元素， 并指向下个节点
class Node:
    def __init__(self, initdata):
        self.data = initdata
        self.next = None
        
    def getData(self):
        return self.data
    
    def getNext(self):
        return self.next
    
    def setData(self, newdata):
        self.data = newdata
        
    def setNext(self, newnext):
        self.next = newnext

# 无序列表
class UnorderedList:
    def __init__(self):
        self.head = None
        
    def isEmpty(self):
        return self.head == None
    
    def add(self, item):
        temp = Node(item)
        temp.setNext(self.head)
        self.head = temp
        
    def length(self):
        current = self.head
        count = 0
        while current != None:
            count += 1
            current = currrent.getNext()
        return count
    
    def search(self, item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getData() == item:
                found = True
            else:
                current = current.getNext()
                
        return found
    
    def remove(self, item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()
                
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())
    
# 有序列表
class OrderedList:
    def __init__(self):
        self.head = None
        
    def isEmpty(self):
        return self.head == None
    
    def add(self, item):
        current = self.head
        previous = None
        stop = False
        while current != None and not stop:
            if current.getData() > item:
                stop = True
            else:
                previous = current
                current = current.getNext()
        temp = Node(item)
        if previous == None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)
        
    def length(self):
        current = self.head
        count = 0
        while current != None:
            count += 1
            current = currrent.getNext()
        return count
    
    def search(self, item):
        current = self.head
        found = False
        stop = False
        
        while current != None and not found and not stop:
            if current.getData() == item:
                found = True
            else:
                if current.getData() > item:
                    stop = True
                else:
                	current = current.getNext()
                
        return found
    
    def remove(self, item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()
                
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())
```





