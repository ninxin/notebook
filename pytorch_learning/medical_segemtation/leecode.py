from typing import List

"""
给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
"""""
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        def backtrack(index, bucket, target):
            # 表示最后一个球都已经放进去了，结果成立
            if index == len(nums):
                return True
            for i in range(k):
                # 如果该桶和前一个桶值相同，则和前面的桶结果一样，所以跳过
                if i > 0 and bucket[i-1] == bucket[i]:
                    continue
                # 如果该桶加入该球之后超过阈值，则跳过
                if bucket[i] + nums[index] > target:
                    continue
                bucket[i] += nums[index]
                if backtrack(index+1, bucket, target):
                    return True
                # 回溯到未添加球的情况，进行下一次循环
                bucket[i] -= nums[index]
            return False

        total = sum(nums)
        if total % k != 0:
            return False
        target = total // k
        n = len(nums)
        nums = sorted(nums, reverse=True)       # 从大到小
        bucket = [0] * k
        return backtrack(0, bucket, target)

"""
对于某些非负整数 k ，如果交换 s1 中两个字母的位置恰好 k 次，能够使结果字符串等于 s2 ，则认为字符串 s1 和 s2 的 相似度为 k 。

给你两个字母异位词 s1 和 s2 ，返回 s1 和 s2 的相似度 k 的最小值。
"""
