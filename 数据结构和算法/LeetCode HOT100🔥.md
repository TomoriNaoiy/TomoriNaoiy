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
# 第二题
<img width="1153" height="1002" alt="image" src="https://github.com/user-attachments/assets/40545940-77eb-4396-8a1a-72b371151a2c" />
其实很简单 但是做不出来因为对字典的了解太差了 还得练
思路是将每个字符排序 然后做为键放入 但是要使用defaludict（list）以确保没有的时候放的是空列表
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash=defaultdict(list)
        for s in strs:
            temp=''.join(sorted(s))
            
            hash[temp].append(s)
        return list(hash.values())

```

# 第五题
<img width="1184" height="1154" alt="image" src="https://github.com/user-attachments/assets/dfab65b9-2c43-445c-b79d-5a865249e736" />
一个双指针的题目 如果暴力会o(n²) 我们用双指针 

在这里有个问题 我用双指针的时候太注重贪心的思想了 导致无法很好的遍历每一个点 实际上 由于我们双指针实际上是一次完整的遍历 而且每次都保留最大值 所以我们指针移动的条件应该给的松一点 不如在这里 我一开始给的条件是如果面积更大才移动 但这样就导致如果前面有更大的就被小的卡住了 应该直接判断高度更小就移动的 这样就可以完整的遍历一遍了 而且最大值也会被保留
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left=0
        right=len(height)-1
        ma=0
        for i in range(len(height)-1):
            res=(right-left)*min(height[left],height[right])
            ma=max(ma,res)
            while height[left]>=height[right] and left<=right:
                right-=1
                res=(right-left)*min(height[left],height[right])
                ma=max(ma,res)
            left+=1
            
            
        return ma

```
