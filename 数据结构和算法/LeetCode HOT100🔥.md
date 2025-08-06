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
# 第七题
<img width="1301" height="880" alt="image" src="https://github.com/user-attachments/assets/c41a6552-2b68-4ba1-90fd-b5432653d8b8" />

非常难绷的一题 硬肝了两个小时无法战胜 看答案才知道思路

讲思路面前 得想知道 一格的蓄水量 取决于右边最大和左边最大中的较小的那一个

思路：每一格的蓄水量等于左边和右边的小格减去当前高度 也就是说如果旁边是1 这个是0 就1格 旁边是3 这边是3 就是0（包括自己）

我们用两个指针 一个从左开始 一个从右开始 并且每一步保留左边右边目前最大值 如果左边小 就用左边减去当前 并且左边前进 如果右边大或者等 右边减一格

**不好想的部分** 为什么离得那么远 也能用那一格的值来比较？因为我们知道 如果右边最大值大 说明右边一定有能够挡住的 不管中间怎么样 因此我们可以这么做 一个个累加起来

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = pre_max = suf_max = 0
        left, right = 0, len(height) - 1
        while left < right:
            pre_max = max(pre_max, height[left])#不管更新最大值
            suf_max = max(suf_max, height[right])
            if pre_max < suf_max:
                ans += pre_max - height[left]
                left += 1
            else:
                ans += suf_max - height[right]
                right -= 1
        return ans


```
# 第八题
滑动窗口模块 实际上和双指针非常像 都是通过left和right指针进行左右移动 区别可能就是滑动窗口是有固定的区间大小或者通过区间大小解决问题 而双指针是通过特点条件从而移动指针解决问题
<img width="1133" height="823" alt="image" src="https://github.com/user-attachments/assets/3906c8cf-0a4c-4a69-9af1-d60de6a69bb4" />
这题要找的是最长不重复部分 如果我们直接遍历一遍 会导致其中有漏掉的部分 因此我们需要的是以每一个元素为起点的遍历 但是又得是o（n） 怎么办呢 我们使用一个集合 如果不在就放入 同时right++
如果在的话 就把left右移 直到删掉第一个和right重复的元素 也就是remove（left） 然后left++

这样可能会有疑问 那么这样怎么保留他最大值呢 所以我们每次left停止右移或者right遇到新的元素时就取一次最大值 这样就能够完美的保存每次的最大值了
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        se=set()
        left=0
        ma=0
        for i,v in enumerate(s):
            while v in se:
                se.remove(s[left])
                left+=1
            se.add(v)
            ma=max(ma,i-left+1)
        return ma
```
这里面有几个细节 首先即使遇到了已经有的 删掉前面的之后还是需要把当前这个放进去 不然就漏掉了对吧 然后每次维护最大值
# 第九题 
同样是滑动窗口 但这一题我觉得我的思路是比较经典的滑动窗口思想 不过使用了sort导致复杂度比较高 好在是过了
思路： 先给i为right指针不断向前 然后在循环里面写一个while维护窗口始终为len（p） 当达到第一个窗口的条件时开始判断 如果满足p就返回left。
与答案的差别就在于我判断异位的方法 我使用的是sort后放入hash字典 答案的方法是通过计算26个字母的数量 很明显他的复杂度小但是不好写 我这把更快但是好写
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        hash=dict()
        hash[''.join(sorted(p))]=0
        ans=[]
        left=0
        for i in range(len(s)):
            while i-left>len(p)-1:
                left+=1
            if i>=len(p)-1:
                if ''.join(sorted(s[left:i+1])) in hash:
                    ans.append(left)
        return ans
```
