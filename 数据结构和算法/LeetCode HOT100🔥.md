算法的学习已经有一点点基础了 在大二开始前先试试能否完成力扣HOT100 提升算法能力！
==============================

1. 两数之和 

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。

 

示例 1：

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
示例 2：

输入：nums = [3,2,4], target = 6
输出：[1,2]
示例 3：

输入：nums = [3,3], target = 6
输出：[0,1]


**思路** 非常经典的数列题目 如果直接暴力 很明显会o(n²) 绝对gg 而前缀和又无法处理这种搜索类问题 因此我们使用哈希表进行优化 

将target-i放入哈希表中 然后寻找到和这个值相等的元素后直接返回 因为是在一次循环中完成的 因此是o（n） 也就是我们俗称的空间换时间
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash = dict()
        for i, num in enumerate(nums):
            if target - num in hash:
                return (hash[target - num], i)
            hash[num] = i
```

