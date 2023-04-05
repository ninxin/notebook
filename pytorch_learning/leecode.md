**给你一个数组，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。**

要求使用空间复杂度为 O(1) 的 原地 算法

```python
# 先将整个数组翻转，再根据右移多少位置将前面k个数据翻转回来，最后把剩下的翻转回来
# [1, 2, 3, 4, 5], 2
# [5, 4, 3, 2, 1]
# [4, 5, 1, 2, 3]
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
     	def reverse(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        n = len(nums)
        k %= n
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
```



**判断数组是否存在重复元素**

```python
# 先排序，在判断相邻的是否相等
# 放到集合中，若长度改变了则说明有重复值
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        # l = len(nums)
        # nums = sorted(nums)
        # for i in range(l-1):
        #     if nums[i] == nums[i+1]:
        #         return True
        # return False  

        # tmp = set(nums)
        # if len(nums) == len(tmp):
        #     return False
        # else:
        #     return True

        return len(set(nums)) != len(nums)
```



**给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。**

进阶：分治法

```python

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            # nums[i-1]为数组前一项最大的子序和
            # 这样每项都是当前长度的最大子序和
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
```



**第n个斐波那契数**

```python
class Solution:
    def fib(self, n: int) -> int:
        # 列出所有的斐波那契数，再选
        num = [1]*31
        num[0] = 0
        for i in range(2, 31):
            num[i] = num[i-1]+num[i-2]
        return num[n]
    	
        # 动态规划
        # 之和前两个数有关，所以用两个位置来存放
        # 前二的被前一的数替代， 前一的数被他们的和代替，这样位置二的数就一直是新的斐波那契数了
        num = [0, 1]
        if n <= 1:
            return num[n]
        for i in range(2, n+1):
            tmp = num[0] + num[1]
            num[0] = num[1]
            num[1] = tmp
        return num[1]
    
    	# 递归，速度巨慢， 2**n复杂度
        if n < 2:
            return n
        return self.fib(n-1)+self.fib(n-2)
```



   **泰伯拿起数**

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        # 同上的方法
        nums = [0, 1, 1]        
        if n <= 2:
            return nums[n]
        for i in range(3, n+1):
            tmp = nums[0]+nums[1]+nums[2]
            nums[0] = nums[1]
            nums[1] = nums[2]
            nums[2] = tmp
        return nums[2]
    
    	# 更简洁
        a, b, c = 0, 1, 1
        for i in range(n) :
            a, b, c = b, c, a+b+c
        return a
    	
        # 不用交换
        nums = [0, 1, 1]
        for i in range(3, n+1):
            nums[i%3] = nums[0] + nums[1] + nums[2]
        return nums[n%3]
```



**给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。**

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # 遍历数组，为零则和后面的非零元素交换
   		l = len(nums)
        j = l-1
        for i in range(l):
            if nums[i] == 0:
                for j in range(i+1, l):
                    if nums[j] != 0:
                        nums[i], nums[j] = nums[j], nums[i]
                        break
		
        # 双指针， 都从头开始。若非零则交换顺序，为零则一个继续前进，一个停在原地，直到找到非零元素交换
        low=0
        for fast in range(len(nums)):
            if nums[fast]!=0:
                # nums[low]=nums[fast]
                nums[low], nums[fast] = nums[fast], nums[low]
                low+=1    
        #不交换，非零就赋值，最后将后面的元素赋值为零
        # n=len(nums[low:])
        # nums[low:]=n*[0]

        # 为零则删除，并再末尾加0
        for i in range(nums.count(0)):
            nums.remove(0)
            nums.append(0)
		
        # 按布尔值降序排序，非零值都视为一样的，所以不改变顺序
        nums.sort(key=bool, reverse=True)
```



**数组中两数之和等于目标值的位置**

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
    	# 双指针，指向两端
        l = len(numbers)
        left, right = 0, l-1
        while left < right:
            sum = numbers[left] + numbers[right]
            if sum > target:
                right -= 1
            elif sum < target:
                left += 1
            else:
                return [left+1, right+1]

        # 键值对
        # 用目标值减去数组中的数，作为字典的键，索引作为值
        # 如果数组中的数在字典的键内，则说明二者相加等于目标值
        dic = {}
        for i in range(len(numbers)):
            if numbers[i] in dic.keys():
                return [dic[numbers[i]] + 1, i + 1]
            else:
                dic[target - numbers[i]] = i
```

**编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数**

```py
class Solution:
    def hammingWeight(self, n: int) -> int:
        #位运算，n & (n-1)可以将n中最后一位的1变为0，统计变为0的次数即可
        cnt = 0 #初始化次数为0
        while n: #循环条件是n不为空
            n &= n - 1
            cnt += 1
        return cnt
```



**假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？**

```python
# 和斐波那契数类似
class Solution:
    def climbStairs(self, n: int) -> int:
        # 到达第k层的次数=到达第k-1层的次数+到达第k-2层的次数
        a = [1, 2]
        if n < 3:
            return a[n-1]
        for i in range(2, n):
            a[0], a[1] = a[1], a[0]+a[1]
        return a[1]
```



**给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。请你计算并返回达到楼梯顶部的最低花费。**

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # 动态规划
        # 数组每个元素为到达这一步的最小花费
        dp = [0] * (len(cost))
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, len(cost)):
            dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]
        return min(dp[len(cost) - 1], dp[len(cost) - 2])
```



**给定一个字符串 `s` ，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。**

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # l, r = 0, 0
        # tmp = list(s)
        # for i in range(len(tmp)):
        #     if tmp[i] == ' ':
        #         r = i - 1
        #         while l < r:
        #             tmp[l], tmp[r] = tmp[r], tmp[l]
        #             l += 1
        #             r -= 1
        #         l = i+1
        #     elif i == (len(tmp) - 1):
        #         r = i
        #         while l < r:
        #             tmp[l], tmp[r] = tmp[r], tmp[l]
        #             l += 1
        #             r -= 1
        # a = ''
        # for j in tmp:
        #     a += j
        # return a   

        l, r = 0, 0
        a = ''
        for i in range(len(s)):
            if s[i] == ' ':
                r = i - 1
                while l <= r:
                    a += s[r]
                    r -= 1
                a += ' '
                l = i + 1
            elif i == len(s) - 1:
                r = i
                while l <= r:
                    a += s[r]
                    r -= 1
        return a
```



**给你两个整数数组 `nums1` 和 `nums2` ，请你以数组形式返回两数组的交集。**

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 先变成集合取交集，得到一样的元素，在找相同元素的个数
        # inter = set(nums1) & set(nums2)
        # l = []
        # for i in inter:
        #     l += [i] * min(nums1.count(i), nums2.count(i))  
        # return l

        # 先排序，再逐一比较
        nums1.sort()
        nums2.sort()
        l1, l2 = 0, 0
        a = []
        while l1 < len(nums1) and l2 < len(nums2):
            if nums1[l1] > nums2[l2]:
                l2 += 1
            elif nums1[l1] < nums2[l2]:
                l1 += 1
            else:
                a.append(nums1[l1])
                l1 += 1
                l2 += 1
        return a
```

给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 0
        a = 0
        for i in range(1, len(prices)):
            if prices[i] < prices[l] and prices[i] < prices[r]:
                l ,r = i, i
                # if prices[r]-prices[l]>a:
                #     a = prices[r]-prices[l]
            elif prices[i] > prices[r]:
                r = i
                if prices[r]-prices[l]>a:
                    a = prices[r]-prices[l]
        return a

```

**给你一个正整数 `num` 。如果 `num` 是一个完全平方数，则返回 `true` ，否则返回 `false` 。**

```python
# 不要从2到num//2遍历
# 二分法，找中间点判断在左边还是在右边
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num == 1:
            return True
        l, r = 2, num//2
        while l <= r:
            mid = (l+r)//2
            if mid*mid > num:
                r = mid-1
            elif mid*mid < num:
                l = mid+1
            else:
                return True
        return False
```

**数组a中有多少数与数组b中所有数的距离大于d**

```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        # a = 0
        # for i in arr1:
        #     for j in arr2:
        #         if i-d <= j and j <= i+d:
        #             a += 1
        #             break
        # return len(arr1)-a

        # 二分法
        # 对数组2排序，再判断
        arr2.sort()        
        a = 0
        for i in arr1:
            l, r = 0, len(arr2)-1
            while l <= r:
                mid = (l+r)//2
                if arr2[mid] < i-d:
                    l = mid+1
                elif arr2[mid] > i+d:
                    r = mid-1
                else:
                    a += 1
                    break
        return len(arr1)-a
```

**给定由一些正数（代表长度）组成的数组 `nums` ，返回 *由其中三个长度组成的、**面积不为零**的三角形的最大周长* 。如果不能形成任何面积不为零的三角形，返回 `0`**

```python
# 先排序，再从后遍历
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        i = 0
        l = len(nums) - 1
        while i < len(nums)-2:
            if nums[l-i] < nums[l-i-1] + nums[l-i-2]:
                return nums[l-i]+nums[l-i-1]+nums[l-i-2]
            else:
                i += 1
        return 0

```

**打家劫舍**

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

```python
# 动态规划
# [5, 1, 1, 5]
# [1, 2, 3, 1]
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 数组元素代表当前累计的最大值，每家都偷
        # [5, 1, 6, 10]
        # [1, 2, 4, 3]
        l = len(nums)
        
        if l == 1:
            return nums[0]
        elif l == 2:
            return max(nums[0], nums[1])
        elif l == 3:
            return max(nums[0]+nums[2], nums[1])

        a = [0] * l
        a[0] = nums[0]
        a[1] = max(nums[0], nums[1])
        a[2] = max(nums[0]+nums[2], nums[1])
        for i in range(3, l):
            a[i] = max(a[i-2], a[i-3]) + nums[i]
        return max(a[l-1], a[l-2])

        # 数组代表最大收益的做法，为了最大收益当前这家可以不偷
        # [5, 5, 6, 10]
        # [1, 2, 4, 4]
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[-1]
```

**上题首尾不能同时偷**

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
          return 0
        if n <= 2:
          return max(nums)
        # 不抢第一个
        dp1 = [0] * n
        dp1[0] = 0
        dp1[1] = nums[1]
        for i in range(2, n):
          dp1[i] = max(dp1[i-1],nums[i] + dp1[i-2])

        # 不抢最后一个
        dp2 = [0] * n
        dp2[0] = nums[0]
        dp2[1] = max(nums[0],nums[1])
        for i in range(2, n-1):
          dp2[i] = max(dp2[i-1],nums[i] + dp2[i-2])
        return max(dp1[n-1],dp2[n-2])
```

给你一个整数数组 nums ，可以选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。

返回你能通过这些操作获得的最大点数。

**选择删除的计入点数，必须删除的不计入点数，可以一直删除直到没有元素， 也是打家劫舍问题**

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        # 一个数组存放相同元素的总和，索引就是元素的值
        a = [0] * (max(nums)+1)
              
        for i in nums:
            a[i] += i

        # 打家劫舍
        l, r = a[0], max(a[0], a[1])
        for j in range(2, len(a)):
            l, r = r, max(l+a[j], r)
        return r
```

**给定一个头结点为 `head` 的非空单链表，返回链表的中间结点。如果有两个中间结点，则返回第二个中间结点。**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        l1, l2 = head, head
        while l1.next:
            if l1.next.next:
                l1 = l1.next.next
                l2 = l2.next
            else:
                l1 = l1.next
                l2 = l2.next
        return l2
```

**给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 栈，先入后出，全进去后再出去n个，再将最后一个的下个节点指向下下个，就删除了第n个
        new = ListNode(0, head)
        a = []
        tmp = new
        while tmp:
            a.append(tmp)
            tmp = tmp.next
        for i in  range(n):
            a.pop()
        a[-1].next = a[-1].next.next
        return new.next

        # 双指针，之间的距离相差n，当一个到达末尾时，另一个指针就到达了要删除的位置，将他的结点指向下下个
        dummy = ListNode(0, head)
        first = head
        second = dummy
        for i in range(n):
            first = first.next

        while first:
            first = first.next
            second = second.next
        
        second.next = second.next.next
        return dummy.next
```

**给定一个字符串 `s` ，请你找出其中不含有重复字符的 最长子串的长度。**

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 一个按顺序遍历，另一个在前一个的基础上遍历，如果有重复值，则停止遍历
        e1, e2 = 0, 0
        l = len(s)
        a = 1
        tmp = []
        if l == 0:
            return 0
        while e1 <= l-1-a:
            while e2 <= l-1:
                if s[e2] not in tmp:
                    tmp.append(s[e2])
                    e2 += 1
                    a = max(a, len(tmp))
                else:
                    tmp = []
                    break
            e1 += 1
            e2 = e1
        return a

        # 遍历一次，有重复值则去掉重复值的第一个，继续遍历
        tmp = []
        a = 0
        for i in s:
            if i in tmp:
                tmp = tmp[tmp.index(i)+1:]
                
            tmp.append(i)
            a = max(a, len(tmp))
        return a
```

给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。

**s1 的排列之一是 s2 的 子串 。**

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        l1, l2 = len(s1), len(s2)
        
        # 判断排序后的s2子串是否和排序后的s1相同
        for i in range(0, l2-l1+1):
            if sorted(list(s1)) == sorted(list(s2[i:i+l1])):
                return True
        return False

        # 类似字典，存放字符和出现的次数
        c1 = collections.Counter(s1)
        i = 0
        while i < l2-l1+1:
            c2 = collections.Counter(s2[i:i+l1])
            if c1 == c2:
                return True
            i += 1

        return False
```

有一幅以 m x n 的二维整数数组表示的图画 image ，其中 image [i][j] 表示该图画的像素值大小。

你也被给予三个整数 sr ,  sc 和 newColor 。你应该从像素 image[sr][sc] 开始对图像进行 上色填充 。

为了完成 上色工作 ，从初始像素开始，记录初始坐标的 上下左右四个方向上 像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应 四个方向上 像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为 newColor 。

最后返回 经过上色渲染后的图像 。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        r = len(image)
        c = len(image[0])
        tmp = image[sr][sc]
        
        if tmp == color:
            return image
        
        # 深度优先搜索
        def DFS(row, col):
            if image[row][col] == tmp:
                image[row][col] = color
                if row - 1 >= 0:
                    DFS(row-1, col)
                if row + 1 < r:
                    DFS(row+1, col)
                if col - 1 >= 0:
                    DFS(row, col-1)
                if col + 1 < c:
                    DFS(row, col+1)

        DFS(sr, sc)
        return image

        # 广度优先搜索
        LIST = []
        def FS(row, col):
            # 将满足条件的放到数组中，以免重复遍历
            if image[row][col] == tmp and (row, col) not in LIST:
                LIST.append((row, col))
                if row - 1 >= 0:
                    FS(row-1, col)
                if row + 1 < r:
                    FS(row+1, col)
                if col - 1 >= 0:
                    FS(row, col-1)
                if col + 1 < c:
                    FS(row, col+1)
                image[row][col] = color
        FS(sr, sc)
        return image
```

给你一个大小为 m x n 的二进制矩阵 grid 。

岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

岛屿的面积是岛上值为 1 的单元格的数目。

计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 广度优先搜索
        r, c = len(grid), len(grid[0])
        tmp = 0
        self.a = 0

        def FS(row, col):
            if new[row][col] == 1:
                # 变回0是为了避免重复计算
                new[row][col] = 0
                self.a += 1
                if row + 1 < r:
                    FS(row+1, col)
                if row - 1 >= 0:
                    FS(row-1, col)
                if col + 1 < c:
                    FS(row, col+1)
                if col - 1 >= 0:
                    FS(row, col-1)
                
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1:
                    new = grid		# 在新的数组中变换
                    self.a = 0
                    FS(i, j)
                    tmp = max(tmp, self.a)
        
        return tmp
```

给你两棵二叉树： root1 和 root2 。

想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。

返回合并后的二叉树。

注意: 合并过程必须从两个树的根节点开始。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        def merge(root1: TreeNode, root2: TreeNode):
            if not root1 and not root2:
                return None;
            if not root1:
                return root2;
            if not root2:
                return root1
            root1.left=merge(root1.left,root2.left)
            root1.right=merge(root1.right,root2.right)
            root1.val+=root2.val;
            return root1;
        return merge(root1,root2);

        if root1 and root2:
            root1.val += root2.val
            root1.left = self.mergeTrees(root1.left, root2.left)
            root1.right = self.mergeTrees(root1.right, root2.right)
            return root1
        else:
            return root1 if root1 else root2
```

给定一个 **完美二叉树** ，其所有叶子节点都在同一层，每个父节点都有两个子节点。

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        # 1.用队列存储每一行节点，先进先出，出来的next指向没出来的第一个
        # 每出来一个，将他的子节点，也就是下一层的入队
        if not root:
            return root
        
        # 初始化队列同时将第一层节点加入队列中，即根节点
        Q = collections.deque([root])
        
        # 外层的 while 循环迭代的是层数
        while Q:
            
            # 记录当前队列大小
            size = len(Q)
            
            # 遍历这一层的所有节点
            for i in range(size):
                
                # 从队首取出元素
                node = Q.popleft()
                
                # 连接
                if i < size - 1:
                    node.next = Q[0]
                
                # 拓展下一层节点
                if node.left:
                    Q.append(node.left)
                if node.right:
                    Q.append(node.right)
        
        # 返回根节点
        return root


        # 2. next的连接方式有两种
        # 一种是同一个父节点，head.left.next = head.right
        # 另一种不是，可以通过上一层的指针找到，head.right.next = head.next.left
        if not root:
            return root
        
        # 从根节点开始
        leftmost = root
        
        while leftmost.left:
            
            # 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
            head = leftmost
            while head:
                
                # CONNECTION 1
                head.left.next = head.right
                
                # CONNECTION 2
                if head.next:
                    head.right.next = head.next.left
                
                # 指针向后移动
                head = head.next
            
            # 去下一层的最左的节点
            leftmost = leftmost.left
        
        return root 
```

请你判断一个 `9 x 9` 的数独是否有效。只需要 **根据以下规则** ，验证已经填入的数字是否有效即可。

行， 列， 小九宫格出现的数字都不能重复

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # 1、先生成三个数组
        rows = [[0] * 9 for _ in range(9)]      # 存放每行不同数字出现的次数，[i][1]第i行5这个数字出现的次数
        columns = [[0] * 9 for _ in range(9)]   # 存放每列不同数字出现的次数，[i][1]第i列5这个数字出现的次数
        subboxes = [[[0] * 9 for _ in range(3)] for _ in range(3)]  # 存放九宫格数字出现的次数
        # 遍历行
        for i in range(9):
            for j in range(9):
                c = board[i][j]
                if c != '.':
                    c = int(c) - 1
                    rows[i][c] += 1
                    columns[j][c] += 1
                    subboxes[int(i/3)][int(j/3)][c] += 1
                    if rows[i][c] > 1 or columns[j][c]>1 or subboxes[int(i/3)][int(j/3)][c]>1:
                        return False
        return True
```

**将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 **

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 递归
        if list1 is None:
            return list2
        elif list2 is None:
            return list1
        elif list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2

        # 迭代
        prehead = ListNode(-1)  # 新的链表

        prev = prehead      # 指针，要变化的
        while list1 and list2:
            if list1.val <= list2.val:
                prev.next = l1
                list1 = list1.next
            else:
                prev.next = l2
                list2 = list2.next            
            prev = prev.next

        # 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev.next = list1 if list1 is not None else list2

        return prehead.next
```

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        # 1.删除节点分为两种情况，删除头节点和非头节点
        while head and head.val == val:
            # 让自head起第一个值不为val的节点作为头节点
            # 退出while循环时，有两种情况
            # 1 head为空(即链表左右节点值均为val，则进入if并return
            # 2 找到了第一个值不为val的节点(是真正的头节点)，那么之后就开始对该节点之后的非头节点的元素进行遍历处理
            head = head.next
        if head is None:
            return head
        node = head
        while node.next:
            if node.next.val == val:
                node.next = node.next.next
            else:
                node = node.next
        return head
       
        # 2.添加一个虚拟头节点，对头节点的删除操作与其他节点一样
        pre_node = ListNode(next=head)
        node = pre_node
        while node.next:
            if node.next.val == val:
                node.next = node.next.next
            else:
                node = node.next
        return pre_node.next

        # 3.递归
        if head is None:
            return head
        head.next = self.removeElements(head.next, val)
        # 利用递归快速到达链表尾端，然后从后往前判断并删除重复元素
        return head.next if head.val == val else head
        # 每次递归返回的为当前递归层的head(若其值不为val)或head.next
        # head.next及之后的链表在深层递归中已经做了删除值为val节点的处理，
        # 因此只需要判断当前递归层的head值是否为val，从而决定head是删是留即可
```

给定一个已排序的链表的头 `head` ， 删除所有重复的元素，使每个元素只出现一次。返回 已排序的链表 。

 ```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None:
            return None

        node = head
        while node.next:
            if(node.val == node.next.val):
                node.next = node.next.next
            else:
                node = node.next
        return head
 ```

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        anchor = head #创建锚节点即头部节点，属于浅拷贝
        while anchor.next:
            temp = anchor.next #锚节点的下一个节点存入temp
            anchor.next = anchor.next.next 
            #删除锚节点的下一个节点，即锚节点的下一个节点向后移动了一位
            #注意同时链表head内对应的节点也已被删除。
            temp.next = head #temp接到head前面
            head = temp #更新head
        return head

        p, rev = head, None
        while p:
            rev, rev.next, p = p, rev, p.next
        return rev
```

给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        Q = collections.deque([])
        visited = set()
        # 初始化队列，将所有起始点加入
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    Q.append((i, j))
                    visited.add((i, j))
        # 将所有相邻节点加入队列
        while Q:
            i, j = Q.popleft()
            for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if 0 <= x < m and 0 <= y < n and (x, y) not in visited:
                    mat[x][y] = mat[i][j] + 1
                    visited.add((x, y))
                    Q.append((x, y))
        return mat
```

在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：

值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。

返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # 广度优先搜索
        x,y,time = len(grid),len(grid[0]),0
        D,queue = [[-1,0],[0,-1],[0,1],[1,0]],[]  #四个方向的坐标和队列
        for i in range(x):
            for j in range(y):
                if grid[i][j] == 2:
                    queue.append((i,j,0))
        while queue:  
            i,j,time = queue.pop(0)
            for d in D:
                loc_i,loc_j = i+d[0],j+d[1]
                if 0 <=loc_i<x and 0<=loc_j<y and grid[loc_i][loc_j]==1:
                    grid[loc_i][loc_j] = 2
                    queue.append((loc_i,loc_j,time+1))      # 每一轮的time都一样
        for g in grid:  
            if 1 in g:
                return -1      
        return time
```

给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 首次出现target的位置
        def left_func(nums,target):
            n = len(nums)-1
            left = 0
            right = n
            while(left<=right):
                mid = (left+right)//2
                if nums[mid] >= target:
                    right = mid-1
                if nums[mid] < target:
                    left = mid+1
            return left
        a =  left_func(nums,target)
        b = left_func(nums,target+1)
        if  a == len(nums) or nums[a] != target:
            return [-1,-1]
        else:
            return [a,b-1]
```

你总共有 n 枚硬币，并计划将它们按阶梯状排列。对于一个由 k 行组成的阶梯，其第 i 行必须正好有 i 枚硬币。阶梯的最后一行 可能 是不完整的。

给你一个数字 n ，计算并返回可形成 完整阶梯行 的总行数。

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        
        # for i in range(2**20):
        #     if (1+i)*i/2 > n:
        #         return i-1

        # 满足条件的第一个数
        left, right = 1, n
        while left < right:
            mid = (left + right + 1) // 2
            if mid * (mid + 1) <= 2 * n:
                left = mid
            else:
                right = mid - 1
        return left

```

给你一个 **严格升序排列** 的正整数数组 `arr` 和一个整数 `k` 。

请你找到这个数组里第 `k` 个缺失的正整数。

```python
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        tmp = []
        for i in range(1, arr[-1]):
            if i not in arr:
                tmp.append(i)           
        if k <= len(tmp):
            return tmp[k-1]
        else:
            return arr[-1] + k - len(tmp)

        # 二分法
        if arr[0] > k:
            return k
        
        l, r = 0, len(arr)
        while l < r:
            mid = (l + r) >> 1
            x = arr[mid] if mid < len(arr) else 10**9
            if x - mid - 1 >= k:
                r = mid
            else:
                l = mid + 1

        return k - (arr[l - 1] - (l - 1) - 1) + arr[l - 1]

        # 拿当前数和上一个数的gap去填k
        prev = 0
        for cur in arr:
            gap = cur - prev - 1
            if k - gap <= 0:
                return prev + k
            k -= gap
            prev = cur
        return cur + k
```

给你一个 `m * n` 的矩阵 `grid`，矩阵中的元素无论是按行还是按列，都以非递增顺序排列。 请你统计并返回 `grid` 中 **负数** 的数目。

```python
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        # return str(grid).count('-')

        res = 0  
        cur = 0
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            top = 0
            end = n - cur  
            while top < end:
                mid = top + (end - top) // 2
                if grid[i][mid] >= 0:
                    top = mid + 1
                elif grid[i][mid] < 0:
                    end = mid 
            cur = n - top 
            res += cur
        return res
```

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 二维的二分
        if matrix is None or len( matrix)==0:
            return False
        row = len( matrix)  #几行
        col = len( matrix[0])   #几列
        l = 0
        r = row * col - 1    #多少个
        while l <= r:
            m = (r+l)//2   #取中
            element = matrix[m//col][m%col]    #//col第几行,%col,这一行第几个
            if element == target:
                return True
            elif element > target:
                r = m- 1
            else:
                l = m+ 1
        return False
```

给你一个大小为 m * n 的矩阵 mat，矩阵由若干军人和平民组成，分别用 1 和 0 表示。

请你返回矩阵中战斗力最弱的 k 行的索引，按从最弱到最强排序。

如果第 i 行的军人数量少于第 j 行，或者两行军人数量相同但 i 小于 j，那么我们认为第 i 行的战斗力比第 j 行弱。

军人 总是 排在一行中的靠前位置，也就是说 1 总是出现在 0 之前。

```python
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # 二分查找
        m, n = len(mat), len(mat[0])
        power = list()
        for i in range(m):
            l, r, pos = 0, n - 1, -1
            # 找到最后一个1的位置
            while l <= r:
                mid = (l + r) // 2
                if mat[i][mid] == 0:
                    r = mid - 1
                else:
                    pos = mid
                    l = mid + 1
            power.append((pos + 1, i))
        # 堆
        heapq.heapify(power)
        ans = list()
        for i in range(k):
            ans.append(heapq.heappop(power)[1])
        return ans

        # 按列遍历，找到第一个为0的行并记录（此时该行后面都置-1），找到k个为止；
        # 如果遍历完不到k个，再按行遍历最后一列，找到!=-1的行号，直到k个。

```

给你一个整数数组 `arr`，请你检查是否存在两个整数 `N` 和 `M`，满足 `N` 是 `M` 的两倍（即，`N = 2 * M`）。

```python
class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        if 0 in arr:
            arr.pop(arr.index(0))
        tmp = [x*2 for x in arr]
        return set(arr) & set(tmp) != set() 

        visited = set()
        for num in arr:
            if num*2 in visited or num/2 in visited:
                return True
            visited.add(num)
        return False
```

给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

你可以按 **任何顺序** 返回答案。

$C_n^m = C_{n-1}^{m-1}+C_{n-1}^m$

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        if k>n or k==0:
            return []
        if k==1:
            return [[i] for i in range(1,n+1)]
        if k==n:
            return [[i for i in range(1,n+1)]]
        # 按公式计算
        answer=self.combine(n-1,k)
        for item in self.combine(n-1,k-1):
            item.append(n)
            answer.append(item)
        
        return answer
```

