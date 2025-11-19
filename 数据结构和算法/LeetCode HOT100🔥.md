算法的学习已经有一点点基础了 在大二开始前先试试能否完成力扣HOT100 提升算法能力！
==============================

# 1. 两数之和 

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

```
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
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        from collections import Counter
        
        ns, np = len(s), len(p)
        if ns < np:
            return []
        
        count_p = Counter(p)
        count_s = Counter()
        res = []
        
        for i in range(ns):
            count_s[s[i]] += 1
            if i >= np:
                if count_s[s[i - np]] == 1:
                    del count_s[s[i - np]]
                else:
                    count_s[s[i - np]] -= 1
            if count_s == count_p:
                res.append(i - np + 1)
        return res
```
可以看一下 和我的思路很像 都是一个窗口 只是他用count计数 离开窗口的就减掉
# 第十题
写的快崩溃的一题 看起来用双指针的思路 但是并不行 看到题解 居然使用的是哈希表加前缀和 万万没想到前缀和还能这么用 直接转化成了第一题 
<img width="1317" height="629" alt="image" src="https://github.com/user-attachments/assets/fcf13a44-5e42-4f23-bc73-71118b4b22e2" />

思路：由于前缀和【i】-[j]=k 可以和第一题两数之和类似 通过哈希表进行存储 一次遍历即可 但是又有不同 这里是次数 而不是只有一个 不能用if ... in 的语法 而是要通过ans+=hash[j-k] 并且每次存入hash[j]+=1 两部 从而在放入每一个值
```python


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        nums_qz = [0 for _ in range(len(nums)+1)]
        for i in range(len(nums)):
            nums_qz[i+1] = nums_qz[i]+nums[i]
        hash = defaultdict(int)
        hash[0] = 1  # 注意：前缀和0出现1次

        for j in range(1, len(nums_qz)):
            # 先统计
            ans += hash[nums_qz[j]-k]
            # 再更新
            hash[nums_qz[j]] += 1
        return ans
```
# 第十一题 
用时最短的一提困难题（其实是在洛谷做过）
<img width="1312" height="887" alt="image" src="https://github.com/user-attachments/assets/0db4e309-f14c-4c9c-953b-a753b73d7b3d" />
由于是固定窗口 又取最大值 用max复杂度超级高 用线段树又太麻烦 我们直接用单调队列
```python
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        que=deque()
        ans=[]
        for i,v in enumerate(nums):
            while que and nums[que[-1]]<=v:
                que.pop()
            que.append(i)
            if que and que[0] <= i-k:
                que.popleft()
            if i>=k-1:
                ans.append(nums[que[0]])
        return ans
```

# 第十二题 
<img width="1244" height="487" alt="image" src="https://github.com/user-attachments/assets/83f9aac3-a6af-49d3-b77d-708b1224555a" />

又是困难题 但是思路其实已经非常简单勒 主要还是一个双指针

难点在于怎么判断包含 这里用的是Counter（）
```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        
        cnt_s=Counter()
        cnt_t=Counter(t)
        left=0
        ans=float("inf")
        a=-1
        b=len(s)-1
        for i,v in enumerate(s):
            cnt_s[v]+=1
            while cnt_t<=cnt_s:
                if i-left<b-a:
                    
                    a=left
                    b=i
                cnt_s[s[left]]-=1
                left+=1
        if a<0:
            return "" 
        return s[a:b+1]
```
如果包含 这里直接使用＞就可以 然后left向前移动 把counter里面的字母-1即可 思路很简单 就是正常的双指针
# 第十三题
其实是一个很简单的题目 又是把序列和转化为前缀和 从n个数转化为2个数的差
<img width="1291" height="790" alt="image" src="https://github.com/user-attachments/assets/6df26df4-7681-486e-baad-7b025cf9faae" />
但是有个问题是 如果直接max和min 则可能出现j在i前面的错误 因此我们必须用两个指针 left始终跟在right后面并且更新为最小值 保证前面减去后面 而且是n的复杂度
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        nums_qz=[0 for _ in range(len(nums)+1)]
        for i in range(len(nums)):
            nums_qz[i+1]=nums_qz[i]+nums[i]
        left=float("inf")
        
        ans=float("-inf")
        left=0
        for i in range(1,len(nums_qz)):
            ans=max(nums_qz[i]-left,ans)
            left=min(left,nums_qz[i])
        return ans
```
# 第十四题
一题看完题解把自己气笑的一题 想了半天怎么去掉里面替换完的区间 又怎么以替换完的区间为坐标来继续判断下一个 这个题解确实太巧妙了 直接使用pop获得元素 然后使用left1，right1，left2和right2来进行区间两端点的更替 简直就是天才 用下标取值的我简直....
<img width="1283" height="686" alt="image" src="https://github.com/user-attachments/assets/507d97d4-baef-4434-90b4-227e8ca05002" />

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        left1,right1=intervals.pop(0)
        ans=[]
        while intervals:
            left2,right2=intervals.pop(0)
            if right1>=left2:
                right1=max(right1,right2)
            else:
                ans.append([left1,right1])
                left1=left2
                right1=right2
        ans.append([left1,right1])
        return ans    
        
```
非常巧妙的通过是否给left1赋值判断是否是同一个区间 然后在不满足同一个区间的时候再往ans里面放 的确非常巧妙

# 第十五题
一个比较新的题目 从前缀和变成前缀积 但是又不同于正常的前缀 题目我们需要找到i的前缀积和后缀积 因此需要和正常的前缀有些区别
<img width="1219" height="387" alt="image" src="https://github.com/user-attachments/assets/75126d5c-da51-4799-be6b-81cff9b723ab" />
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n=len(nums)
        pre=[1 for _ in range(n)]
        suf=[1 for _ in range(n)]
        ans=[]
        for i in range(1,len(nums)):
            pre[i]=pre[i-1]*nums[i-1]
        for i in range(len(nums)-2,-1,-1):
            suf[i]=suf[i+1]*nums[i+1]
        for p,v in zip(pre,suf):
            ans.append(p*v)
        return ans
```
这里的前缀和正常的区别 nums=【1， 2，3，4】
那么pre=[1,1, 2,6]
suf=[24,12,4,1]
对于正常的前缀积是[1,1,2,6,24]
后缀【24,24,12,4,1】
很明显 这里的写法少了一个位置 去掉了最后一位 这样恰好可以满足i的左边和右边的积
# 十六题
非常简单的困难题 思路比较简单 但是或许是一个可以延申的思想
<img width="968" height="812" alt="image" src="https://github.com/user-attachments/assets/ffb7b26e-2903-401b-a2c2-e0338129641a" />
正常排序完之后 普通的思路（我的）就是遍历这个数组 我希望是先把《=0的排除了 然后让num[i]==i的方式来实现 但是这样有很多问题 比如重复 比如有0和-1导致下标和值不一定对其 又不方便用字典来做 而题解使用的是一种动态指针的思路 我的思路是固定的i对应固定的下标 但是题解的思路是给一个i 如果满足这个整数了 就i++ 这样完美解决了重复的问题（因为是遍历） 然后也解决了下标不对其的问题 延申的思路就是 我们不用固定的指标 而是通过动态指标 满足条件则进行变化 以满足需求 是一个很好的思想
```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums.sort()
        y=1
        for i in nums:
            if i == y:
                y+=1
            elif i>y:
                return y
        return y
     
```
# 第十八题 
非常简单的矩阵题 记录一下写的思想
<img width="1215" height="1061" alt="image" src="https://github.com/user-attachments/assets/c6b12963-0e7d-42ec-8cb5-ebb1fbe9fded" />
最简单的肯定是开一个一样大的空间 但是不合适 我使用的方法是构造一个row和一个wow 一个记录行 一个记录列 有出现0 就给那一个赋值1 然后再遍历一遍数组 如果row或者wow中有出现1 就全部赋值0 否则保留 这样的空间消耗小一点
```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rl=len(matrix)
        wl=len(matrix[0])
        row=[0 for _ in range(rl)]
        wow=[0 for _ in range(wl)]
        for i in range(rl):
            for j in range(wl):
                if matrix[i][j]==0:
                    row[i],wow[j]=1,1
        #print(row,wow)
        for i in range(rl):
            for j in range(wl):
                if row[i] or wow[j]:
                    matrix[i][j]=0
```
# 第十九题
一题比较难的矩阵 以前也做过 但还是不会
思路就是 使用四个指针 在循环中巧妙的变化 完成循环
<img width="1241" height="1013" alt="image" src="https://github.com/user-attachments/assets/4ccc601b-58ec-4da4-a25f-4c0e04a6ccc3" />
```python
class Solution:
    def spiralOrder(self, matrix):
        res = []
        if not matrix:
            return res
        
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while left <= right and top <= bottom:
            # 从左到右
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            top += 1
            
            # 从上到下
            for i in range(top, bottom + 1):
                res.append(matrix[i][right])
            right -= 1
            
            # 从右到左
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    res.append(matrix[bottom][i])
                bottom -= 1
            
            # 从下到上
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    res.append(matrix[i][left])
                left += 1
        
        return res
```
# 第二十题
<img width="1243" height="1180" alt="image" src="https://github.com/user-attachments/assets/a605e15d-6e2d-43c6-bd85-66dfebe8b57f" />

旋转图像
做法很简单 但是要知道技巧
逆旋转90 转置+列倒叙
顺90 转置+行倒序
180 行+列倒序
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l=len(matrix)
        r=len(matrix[0])
        for i in range(l):
            for j in range(i,r):
                temp=matrix[i][j]
                matrix[i][j]=matrix[j][i]
                matrix[j][i]=temp
        for i in range(l):
            for j in range(r//2):
                temp=matrix[i][j]
                matrix[i][j]=matrix[i][r-j-1]
                matrix[i][r-j-1]=temp
```
# 二十一题 
矩阵的搜索 很有意思的题目 从中能发现一些思路和技巧
<img width="1304" height="1171" alt="image" src="https://github.com/user-attachments/assets/bc490a9c-4606-4389-8bae-92db9d0c2a6e" />
第一种方法 直接二分 复杂度mlogn
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
      for row in matrix:
        idx = bisect.bisect_left(row,target)
        if idx<len(row) and row[idx]==target:
          return True
      return False
```
bisect.bisect_left(list,target)
通过二分寻找第一个等于target的下标 如果都没有 就返回第一个大于他的下标 如果都小于 就返回列表长度**注意二分是一定要排序的！！！！！！！！**
第二种方法 复杂度o(n+m)
我感觉很难想出来 
每一行的最后一位都是最大 那我们从第一行最后一个开始 如果小 x-=1 如果大 就y+=1 这样就避免了同时考虑要向下和向左的情况 确实很有意思
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        l=len(matrix)
        w=len(matrix[0])
        a=0
        b=w-1
        while b>=0 and a<l:
            if matrix[a][b]==target:
                return True
            if matrix[a][b]<target:
                a+=1
            else:
                b-=1
        return False


```
# 第二十七题
合并链表 有一些难度 关键是掌握方法
<img width="1251" height="1115" alt="image" src="https://github.com/user-attachments/assets/fbeef0a2-59a5-42b8-80a4-5634b8d15a6c" />
做法是 先来一个新的listnode 值多少无所谓 然后同时遍历两个链表 如果那边小 新的头就指向它 然后list要往下 最后新的头也往下 就这样一个一个遍历下去 知道两个中有一个为None 停下来了 但是有剩下的l没有结束 那么直接把头指向剩下的那个即可 最后返回新的头的next即可
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        pre=ListNode(-1)
        a=pre
        while list1 and list2:
            if list1.val<=list2.val:
                pre.next=list1
                list1=list1.next
                
            else:
                pre.next=list2
                list2=list2.next
            pre=pre.next
        pre.next=list1 if list1 is not None else list2
        return a.next
```
# 第二十八题 
没啥好讲的 纯模拟 说一个好玩的点
如果一个数和另一个数位数不同 我们最后要相加（列表里面） 我们可以直接把他变成str 然后用join连接 再变成int直接相加就行 不用*10的n次方这么麻烦了

# 第二十九题
要求是两两交换 难点就在于 如何确保两个 两个进行交换呢
题解的思路很巧妙 每次要执行的链表通过temp来定义 而temp每次有相当于往前两步
<img width="1242" height="811" alt="image" src="https://github.com/user-attachments/assets/7479ca58-8922-42df-9f38-e4da933ec5ad" />
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dum=ListNode(-1)
        dum.next=head
        temp=dum
        while temp.next and temp.next.next:
            node1=temp.next
            node2=temp.next.next
            temp.next=node2
            node1.next=node2.next
            node2.next=node1
            temp=node1
        return dum.next

```
# 第三十题
目前来说感觉最难的一题（当然用数组模拟就太简单了）
<img width="1299" height="1163" alt="image" src="https://github.com/user-attachments/assets/a82f9a83-61e4-409e-abca-70bfda0c027c" />
在了解了如何翻转链表后 还是倒在了链之间的连接中
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
       
        p=head
        cut=0
        while p:
            p=p.next
            cut+=1
        p0=ListNode(next=head)
        dummt=p0
        pre=None
        cur=head
        nex=None
        for _ in range(cut//k):
           

            for _ in range(k):
                nex=cur.next
                cur.next=pre
                pre=cur
                cur=nex
            nex=p0.next
            nex.next=cur
            p0.next=pre
            p0=nex
        
        return dummt.next

```
翻转思路为  给一个pre 一个cur 一个next（防止丢失） 然后遍历一遍链表 让cur指向pre 然后cur和pre不断向前（将链表倒叙连接）
而这里是片段 所以多了一个片段的连接 有点难理解
# 三十一题
<img width="1343" height="1136" alt="image" src="https://github.com/user-attachments/assets/dde67f2f-c5a3-492b-8fda-c7d8be2cb1e4" />

一个对加入了random指针的链表的深度复制 要求不能指向原链表

最大的难点在于怎么对应原链表random的指针

这里使用的是 hash表 将原链表做为键 复制为值 然后一一对应就可以了 很神奇

有一个要注意的点 就是如果random指针为空（None） 就会报错 因为没有这个东西 我们要提前设置**hash[None]=None**！！
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        
        has={None:None}
        if not head:
            return None
        q=head
        while q:
            has[q]=Node(q.val)
            q=q.next
        q=head
        while q:
            has[q].next=has[q.next]
            has[q].random=has[q.random]
            q=q.next
        return has[head]

```
# 第三十四题
纯手搓的困难题 有点成就感 但是不是最优解 最优解居然是使用堆 看来得学习一下也用法了
<img width="1010" height="1066" alt="image" src="https://github.com/user-attachments/assets/9af8e9a7-8b64-4752-b576-b57e5d043aff" />
我自己做的思路就是 每两个进行一次链表合并 就是从守卫开始添加较小的部分 然后完成这两个的排序后 把指针会到开头 然后对下一个链表进行合并
而题解的思路则是把所有已排的链表添加到堆里面 每次找出最小堆 然后把这个最小的next的放入堆中 再次寻找最小堆 直到结束
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        dummy=ListNode(-1)
        temp=dummy
        if lists:
            head=lists[0]
            if len(lists)>1:
                for i in range(1,len(lists)):
                    p=lists[i]
                    while head and p:
                        if head.val<=p.val:
                            temp.next=head
                            head=head.next
                        else:
                            temp.next=p
                            p=p.next
                        temp=temp.next
                    temp.next = head if head is not None else p
                    head=dummy.next
                    temp=dummy
            else:
                return lists[0]
            return dummy.next
                
            
            
```
## 堆算法
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        cur = dummy = ListNode()
        h = [(head.val, i) for i,head in enumerate(lists) if head]
        heapify(h)
        while h:
            v, i = heappop(h)
            cur.next = ListNode(v)
            cur = cur.next 
            if lists[i].next:
                lists[i] = lists[i].next
                heappush(h, (lists[i].val, i))
        return dummy.next

        
```
# 第三十五题
<img width="1234" height="1149" alt="image" src="https://github.com/user-attachments/assets/a42b881a-0a3b-485b-ad51-a3866f3f65e4" />

最久未使用删除的类型 实际上思路就是每次执行后将这个东西放到最上方 由于如果要o（1） 而且我不想写双向链表  所有这里用列表模拟 复杂度高一些 但也能过就是了 要注意的是remove没有返回值了...
```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity=capacity
        self.hash=dict()
        self.use=[]
        self.cut=0
    def get(self, key: int) -> int:
        if key in self.hash:
            self.use.remove(key)
            self.use.append(key)
            return self.hash[key]
            
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.hash:
            self.hash[key]=value
            self.use.remove(key)
            self.use.append(key)
        else:
            self.hash[key]=value
            self.cut+=1
            self.use.append(key)
            if self.cut>self.capacity:
                self.cut-=1
                temp=self.use.pop(0)
                del self.hash[temp]
                

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```
# 第三十七题
一个很简单的深度问题 太久没写递归有点忘了 回顾一下
<img width="1171" height="1111" alt="image" src="https://github.com/user-attachments/assets/3b948f48-7c01-429a-89c8-9d6620f89687" />
```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
```
这里要求最大深度 很明显可以使用dfs

递归怎么来书写呢

直接使用max 会以深度优先（后序）不断向下 如果没有了就返回0 否则返回左或右的最大值+1（深度）
# 第三十八题
一题ac的莫名奇妙的题 自己都还没完全搞懂就结束了 
<img width="1142" height="1111" alt="image" src="https://github.com/user-attachments/assets/2f79089b-a016-4f00-8198-3991fc2dd95c" />
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            temp=self.invertTree(root.left)
            root.left=self.invertTree(root.right)
            root.right=temp
        return root
```
感觉写递归的题目有种“你尽管写最简单的思路 后续的完成交给奇迹”
这里的写法直接是将左右交换的基础语法 递归会自动深入并完成所有内容
# 第三十九题
<img width="1274" height="907" alt="image" src="https://github.com/user-attachments/assets/2d567e4a-7280-4724-add6-2456fbb75040" />

对称树 难度比上一题难 它要同时满足三个条件 左右等 左左等于右右 左右等于右左 因此我们可以使用两个节点而不是单个节点分散 并且return只需要同时三个条件即可
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def check(self,p,q):
        if not p or not q:
            return p == q#如果有null节点
        return p.val==q.val and self.check(p.left,q.right) and self.check(p.right,q.left) #只需要满足这个最大的条件 就可以返回True
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self.check(root.left,root.right)


```
# 第四十题
很明显难于其他简单题 这题已经是树上dp的高端思想了 目前的我无法完全理解 先写下来
<img width="1361" height="1048" alt="image" src="https://github.com/user-attachments/assets/7b118beb-c8a0-403e-8cb8-1ffd5f753d29" />
思路就是 获得拐弯节点的左右链长度之和 然后遍历每个节点 并且对ans进行更新
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans=0
        def check(r):
            if not r:
                return -1
            ll=check(r.left)+1
            rl=check(r.right)+1
            nonlocal ans
            ans=max(ans,ll+rl)
            return max(ll,rl)
        check(root)
        return ans
```
研究了一下 发现它巧妙的结合了上一题求最大深度的需求 通过递归 然后max左右中最大的那一部分 让后最后加上1 就是左（右）的最大深度了 然后再相加 就是直径了 联系之紧密让我叹为观止
# 第四十一题
二叉树的层序遍历 从递归一下进入递推 确实不太适应 而且思路有些新奇 值得学习
<img width="1261" height="1130" alt="image" src="https://github.com/user-attachments/assets/22fc1263-9e46-4e9f-a1d0-eea80a921ab8" />
思路就是使用队列（保存每一层的节点）但是并不是使用while queue这样的条件 而是通过for —— in range（len）的条件 因为循环的同时要加入下一层的左右节点
```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res=[]
        a=deque()
        a.append(root)
        while a:
            temp=[]
            for _ in range(len(a)):
                node=a.popleft()
                temp.append(node.val)
                if node.left:
                    a.append(node.left)
                if node.right:
                    a.append(node.right)
            res.append(temp)
        return res
                


```
# 第四十二题
一个二分+递归构造树的典型 这里是构造一个二叉搜索树
<img width="1164" height="1090" alt="image" src="https://github.com/user-attachments/assets/4ea5cac4-8d1b-4054-81d8-13a87e2e02cf" />
还是递归最基本的思路 由于列表已经排序 我们只需要把列表分成左右和中间 中间为根节点 左右分别用递归获得 然后左右根的左右节点
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        l=len(nums)
        node1=self.sortedArrayToBST(nums[:l//2])
        node2=self.sortedArrayToBST(nums[(l//2)+1:])
        return TreeNode(nums[l//2],node1,node2)

```
**注意** 返回的就是单纯一个节点 不是列表
# 第四十三题
有点陷阱在里面 很恶心 要求的是上面的永远大于下面的（小于 ）因此不能用节点来递归 必需使用数值来递归（不然只会判断一个小子树 而不能保证上面的树一定大于下面的）
<img width="1419" height="1077" alt="image" src="https://github.com/user-attachments/assets/c92c653e-8225-43d7-9ab7-cc6d37484375" />
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode],left=-inf,right=inf) -> bool:
        if not root:
            return True
        return left<root.val<right and self.isValidBST(root.left,left,root.val) and self.isValidBST(root.right,root.val,right)

```
# 第四十四题
很简单的思路 一个二叉树的搜索 要知道的点就是

**只需要中序遍历一个二叉搜索树就可以实现从小到大的排序**
# 第四十六题
<img width="1248" height="913" alt="image" src="https://github.com/user-attachments/assets/1a1da63e-f22d-4ce0-a04e-1de0c07ef018" />
将树转化为链表 

难点在于 这里并不使用next 而是变成了right 也就是多了一个迭代的过程 在前序遍历的过程中 我卡在了究竟是root.right=root.right.left
还是root.right=root.right.right
因为在一个函数中无法完成区分 因此这里思路是 通过使用每个函数中的root
也就是q.right=root这样就不用考虑究竟是left还是right了 

由此学到的一个思路就是 可以通过递归函数的输入值做为一个迭代数（正常思路是将返回值做为数）

这里的递归函数的作用仅仅是进入下一层而已
```python
class Solution:
    def __init__(self):
        self.k=1
        self.dummy=TreeNode(-1)
        self.p=self.dummy
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
       
        if root:
            temp1=root.left
            temp2=root.right
            self.p.right=root
            self.p.left=None
            self.p=root
            self.flatten(temp1)
            self.flatten(temp2)
        
```
# 第四十七题
通过前序和中序遍历构造二叉树 跟之前二分法构造二叉树的思路有异曲同工之妙 但是这里难度更高 因为递归的过程有两个列表的迭代（也就是切片）
<img width="1208" height="1093" alt="image" src="https://github.com/user-attachments/assets/9f183c64-0a3c-4957-a247-005d97599863" />
思路就是 前序遍历中的0是根 然后在中序中找到这个值的下标 那么他左边就是左子树 右边就是右子树 这样又回到了递归的子问题思想 就可以用递归解决
然后每次递归返回切片后的pro和ino列表 最后返回pro【0】，left，right做为一个节点 就完成了递归构造二叉树 和之前的二分非常相似 只是这里的切片就并非中点而是需要主导index寻找
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if preorder:
            key=inorder.index(preorder[0])
            left=self.buildTree(preorder[1:1+key],inorder[:1+key])
            right=self.buildTree(preorder[key+1:],inorder[key+1:])
            return TreeNode(preorder[0],left,right)
```
# 第是四十八题
<img width="1245" height="1137" alt="image" src="https://github.com/user-attachments/assets/f3933358-af54-440d-baff-f08da43b2f9b" />
树上前缀和问题

可以类比之前的序列和为k的问题 思路是通过计算前缀和 然后用哈希表 加上[s-targrt]如果没有就是0 有就会加上个数 然后[s]+=1 把当前节点加上

在这里思路是一致的 只是变成了树上问题 那么就牵扯到两个 一个就是dfs 一个就是回溯
我们使用前缀和无法保证回到另一条线的时候没有删掉先前的部分 因此在+=1完成之和 进行两次dfs 完成后回溯 把s-=1 这就是总体方案
```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        ans=0
        cnt=defaultdict(int)
        cnt[0]=1
        def dfs(root,s):
            if root:
                nonlocal ans
                s+=root.val
                ans+=cnt[s-targetSum]
                cnt[s]+=1
                dfs(root.left,s)
                dfs(root.right,s)
                cnt[s]-=1
        dfs(root,0)
        return ans
```

# 第四十九题
一个寻找最小祖先的递归问题 和之前构造邻接矩阵不同 这里给的是已经有的树 需要我们通化递归获得最小祖先 思路就是 递归寻找左右子树 如果左右都有 那么root就是祖先 如果只有左边有 那么第一个左边就是祖先 同理 第一个右边是祖先 然后如果找到了p或者q或者None就返回root

<img width="1295" height="1125" alt="image" src="https://github.com/user-attachments/assets/4b7108f0-e4fd-46e1-8161-60b8332638c2" />
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if (not root) or (root==p) or (root==q):
            return root
        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)
        if left and right:
            return root
        if left:
            return left
        return right
```
# 第五十题
几乎是完全自己写的苦难题了 还是挺有成绩感的 思路和之前直径一样 但是区别于前缀和的那题 前缀思路是dfs递归然后取值并且回溯 而这里的思路则是对于每个节点分别取左右子链 然后每次取最大值 并进行更新 有一个易错点就是都是负数的时候要单独考虑 返回值返回0就行（相当于不取左右子链 只在意自己）
<img width="1288" height="1195" alt="image" src="https://github.com/user-attachments/assets/67f252d1-dcf0-4cb0-8bbc-5809fac745de" />
```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans=-inf
        def dfs(root,s):
            if not root:
                return 0
            nonlocal ans
            
            ll=dfs(root.left,s)
            rl=dfs(root.right,s)
            s+=root.val    
            ans=max(ans,ll+rl+root.val,root.val,root.val+ll,root.val+rl)
            return max(s+max(ll,rl),0)
        dfs(root,0)
        return ans
```
# 第五十一题
是一个做过的题目 只需要dfs把所有1变成 然后寻找所有都是1的部分 每次ans+1就可以 已经可以自己做出来了~
<img width="1248" height="932" alt="image" src="https://github.com/user-attachments/assets/6c3df9a6-f976-4f2f-aaa9-487abee9cfb1" />
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        ans=0
        def dfs(x,y):
            if grid[x][y]=="1":
                grid[x][y]="2"
                for i,j in [(1,0),(-1,0),(0,-1),(0,1)]:
                    if x+i>=0 and x+i<len(grid) and y+j>=0 and y+j<len(grid[0]):
                        dfs(x+i,y+j)
            
        for a in range(len(grid)):
            for b in range(len(grid[0])):
                if grid[a][b]=="1":
                    dfs(a,b)
                    ans+=1    
        return ans
```
# 第五十二题
一个bfs的题目 大体思路已经能完成了 但是有一点小问题 就是bfs跟递归不一样 不能在函数中调用函数 而是类似与递推的的过程 在循环中每次改变grid 而非函数递归
![Uploading 54de8f80933dff6d50fba502cda36bc8.png…]()
```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        queue=deque()
        ans=-1
        num=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==2:
                    queue.append((i,j))
                if grid[i][j]==1:
                    num+=1
        if num==0:
            return 0
        while queue:
            
            ans+=1
            for _ in range(len(queue)):
                a,b=queue.popleft()
                for p,q in [(1,0),(0,1),(-1,0),(0,-1)]:
                        if a+p>=0 and a+p<len(grid) and b+q>=0 and b+q<len(grid[0]):
                            if grid[a+p][b+q]==1:
                                grid[a+p][b+q]=2
                                num-=1
                                queue.append((a+p,b+q))
        
        return ans if num==0 else -1
```
# 第五十三题
这题很明显是一个拓扑排序 什么是拓扑排序呢 就是一个有向无环图中 使其排序为前驱-》后继的线性排序 也就是确保依赖关系（入度少的在前面）

在这一题中 很明显所有前者都是后继 因此我们通过拓扑排序 分别将入度和图存入列表和字典中 在这里 先通过bfs 每次学习入度为0的点 也就是通过队列 将入度为0的点入列 然后每次把这些点学习后 num-=1 再把后继的入度减掉 再放入入度为0的点 知道队列空 看是否全部学习 不然就返回false（入度只跟前后级有关系 和其他部分无关）
<img width="1254" height="913" alt="image" src="https://github.com/user-attachments/assets/918cd5f9-e064-4ad1-a00a-0c28fa279005" />
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegree=[0 for _ in range(numCourses)]
        ans=defaultdict(list)
        for cur,pre in prerequisites:
            indegree[cur]+=1
            ans[pre].append(cur)
        q=deque()
        for i in range(numCourses):
            if indegree[i]==0:
                q.append(i)
        while q:
            for i in range(len(q)):
                pre=q.popleft()
                numCourses-=1
                for cur in ans[pre]:
                    indegree[cur]-=1
                    if indegree[cur]==0:
                        q.append(cur)
        return numCourses==0
```
# 第五十四题 
前缀树 实际上就是字典树 用于查找是否前缀

做法就是做一个链表 每一个链表都有26个方向（实际上做成字典就行） 然后每个字符依次遍历 从根开始 每次走向字典从存储的方向（字典中存的都是node节点 都有字典） 如果字典中找不到这个方向 说明不是前缀 要么insert插入 要么返回False
<img width="1219" height="1133" alt="image" src="https://github.com/user-attachments/assets/b23f0952-1174-4d7e-8ce5-97658f1b770e" />
```python
class node:
    def __init__(self):
        self.son={}
        self.end=False
class Trie:

    def __init__(self):
        self.root=node()

    def insert(self, word: str) -> None:
        cur=self.root
        for c in word:
            if c not in cur.son:
                cur.son[c]=node()
            cur=cur.son[c]
        cur.end=True
    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.son:
                return False
            cur=cur.son[c]
        return True if cur.end else False

    def startsWith(self, prefix: str) -> bool:
        cur=self.root
        for c in prefix:
            if c not in cur.son:
                return False
            cur=cur.son[c]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```
# 第五十五题
<img width="1108" height="704" alt="image" src="https://github.com/user-attachments/assets/0b296af4-8ca6-4f12-b639-e2df7331d28a" />
回溯题 

讲到回溯 其实dfs根回溯非常相似 都是一条路走到黑 然后退回第一个有分支的路 再走下去 再退回 而回溯和dfs的最大区别就是 回溯多了一个状态重置的过程 因此他们用来应对的题目不同 dfs是遍历 而回溯是搜索

这一题的思路是给一个temp列表 如果长度和nums相同就直接加入ans中 并且结束这一次递归 如果少的话 就进入一个循环 把还没用过的元素依次加入temp中 然后给他已用过的标记 然后递归 这时都是已经被标记的 就会继续寻找没被标记过的 然后完成后回溯 先把这个踢出去 然后把这个元素使用过的标签改为False 再进入下一次循环 就会用下一个数 然后再寻找没用过的数 以此类推
```python
class Solution:
    def __init__(self):
        self.ans=[]
    def permute(self, nums: List[int]) -> List[List[int]]:
        l=len(nums)
        used=[False for _ in range(l)]
        path=[]
        def dfs():
            if len(path)==l:
                self.ans.append(path.copy())
                return
            for i,j in enumerate(nums):
                if used[i]==True:
                    continue
                path.append(j)
                used[i]=True
                dfs()
                path.pop()
                used[i]=False
        dfs()
        return self.ans
```
# 第五十六题
<img width="1171" height="673" alt="image" src="https://github.com/user-attachments/assets/30b08ec3-c345-4fc0-809f-5681e34e5f5b" />
又是回溯题 和上面那题很像 但又有区分 这一题的思路是 每次递归都加入ans 但是函数多了一个参数j 每次j+1 （之前是都要从头遍历 因为都是l长度） 这样就会先把每一个数字放入 然后到最后一个的时候 pop掉 回到倒数第二个函数 然后进行循环 放入循环的下一个数 放完后又pop 然后进入倒数第三个循环 再循环放入下一个数 因此复杂度也是o（n！）

**注意**再ans里面append列表的时候 一定是copy的 因为直接放列表的话 相当于放入地址 之后改变的时候会一起改变
```python
class Solution:
    def __init__(self):
        self.ans=[]
    def subsets(self, nums: List[int]) -> List[List[int]]:
        temp=[]
        def dfs(i):
            self.ans.append(temp.copy())
            for j in range(i,len(nums)):
                temp.append(nums[j])
                dfs(j+1)
                temp.pop()
        dfs(0)



        return self.ans
```
# 第五十六题
<img width="1191" height="1044" alt="image" src="https://github.com/user-attachments/assets/ffb28eb9-d128-4b2e-9699-3957e1e11271" />

完完全全自己做出来的！！爽了

还是经典回溯 但是这题需要用到两次遍历 第一次还是正常的通过传入i参数进行递归 然后在每个数字中进行遍历 遍历的同时进行回溯 就能完成 
```python
class Solution:
    def __init__(self):
        self.ans=[]
        self.dc={"2":["a","b","c"],"3":["d","e","f"],"4":["g","h","i"],"5":["j","k","l"],"6":["m","n","o"],"7":["p","q","r","s"],"8":["t","u","v"],"9":["w","x","y","z"]}
    def letterCombinations(self, digits: str) -> List[str]:
        if digits=="":
            return []
        s=[]
        def dfs(i):
            if len(digits)==len(s):
                self.ans.append(''.join(s))
                return
            for j in range(i,len(digits)):
                for k in self.dc[digits[j]]:
                    s.append(k)
                    dfs(j+1)
                    s.pop()
        dfs(0)
        return self.ans

```
# 第五十七题
又是自己做出来的回溯题 其实类型都差不多了 但时这一题多了一个减枝以及一个小判断 
<img width="1240" height="982" alt="image" src="https://github.com/user-attachments/assets/c48e3a74-e465-41c7-b39c-fcf8b86f074b" />
这题和之前的区别是 之前如果是不重复的（子集） 每次递归要返回一个j+1也就是进入下一个 如果是排序就不用 但是不能出现重复 因此使用bool进行判断 这一题则完全可以重复 但是是集合不能有顺序不同的相同元素 因此我们也加一个小判断（以大小判断） 并且返回一个s和 并且进行一个剪枝 就能完成了
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans=[]
        temp=[]
        #candidates.sort()
        s=0
        def dfs(s):
            #nonlocal s
            if s>target:
                return
            if s==target:
                ans.append(temp.copy())
                return
            for i in range(len(candidates)):
                if (not temp) or (temp and candidates[i]>=temp[-1]):
                    s+=candidates[i]
                    temp.append(candidates[i])
                    dfs(s)
                    temp.pop()
                    s-=candidates[i]
        dfs(0)
        return ans
```
# 第五十九题
括号添加题 我之前把他想复杂了 实际上构造的时候（不是检验）只需要满足左括号数量等于右括号数量就可以了（先放左后放右）
用回溯来做同样是如果left《n 就不断放左 然后如果大于n了 开始判断left和right的关系 开始放右括号 然后一个个减掉 右回到最开始 然后第二阶段判断左和右的关系 在进行一次循环 以此类推
一开始是（（（（（）））））

第二阶段是（）（（（（（）））

然后（）（）（（（）））

<img width="1253" height="725" alt="image" src="https://github.com/user-attachments/assets/6ac61896-b53e-42eb-96fa-16ff53b5e1f5" />
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans=[]
        temp=[]
        
        def dfs(left,right):
            if len(temp)==2*n:
                ans.append(''.join(temp))
                return
            if left<n:
                temp.append("(")
                dfs(left+1,right)
                temp.pop()
            if left>right:
                temp.append(")")
                dfs(left,right+1)
                temp.pop()
        dfs(0,0)
        return ans

#(((()
```
# 第六十题 
一个dfs加回溯题 之前遇到但是不会写 现在以及可以独立完成了 还是有很不错的提升
<img width="1151" height="1144" alt="image" src="https://github.com/user-attachments/assets/6b3030b8-6e83-42db-b532-e689971ec486" />
思路就是正常dfs 如果符合 就下一次dfs 并且index+1

然后外界寻找首位开始递归

回溯的部分通过标记（1） 然后递归完又变回0
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        grid=[[0 for _ in range(len(board[0])+1)] for _ in range(len(board)+1)]
        flag=False
        def dfs(x,y,index):
            nonlocal flag
            for i,j in [(-1,0),(0,1),(1,0),(0,-1)]:
                if index<len(word) and x+i>=0 and x+i<len(board) and y+j>=0 and y+j<len(board[0]) and board[x+i][y+j]==word[index] and grid[x+i][y+j]==0:
                    grid[x+i][y+j]=1
                    #print(board[x+i][y+j],index)
                    if index==len(word)-1:
                        flag=True
                        return
                    dfs(x+i,y+j,index+1)
                    grid[x+i][y+j]=0
        for a in range(len(board)):
            for b in range(len(board[0])):
                if flag:
                    return flag
                if len(word)==1:
                    if board[a][b]==word[0]:
                        flag=True
                if board[a][b]==word[0]:
                    grid[a][b]=1
                    dfs(a,b,1)
                    grid=[[0 for _ in range(len(board[0])+1)] for _ in range(len(board)+1)]
        return flag
```
# 第六十一题
又是回溯 但是以及基本掌握了 通过这几题 大概掌握了回溯的一个用途
<img width="1353" height="1035" alt="image" src="https://github.com/user-attachments/assets/e0f973b2-fe49-4766-b21d-46623429f27e" />
```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans=[]
        temp=[]
        def dfs(index):
            #print(index)
            if index==len(s):
                ans.append(temp.copy())
                return            
            for i in range(index,len(s)):

                t=s[index:i+1]
                if t==t[::-1]:
                    temp.append(t)
                    dfs(i+1)
                    temp.pop()
        dfs(0)
        return ans
```
**遇到这种子集 排列/组合 前缀类型的题目（需要反复遍历的类型）就使用回溯法**

#  六十六题 
<img width="1069" height="954" alt="image" src="https://github.com/user-attachments/assets/f46e96e2-bd0f-416d-9b1d-0d817e801bfa" />
二分 但是局部单调

思路是  在二分的同时寻找单调的部分 然后在单调的部分进行二分

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left,right=0,len(nums)-1
        mid=(left+right)//2
        while left<=right:
            mid=(left+right)//2
            if nums[mid]>=nums[left]:
                if target==nums[mid]:
                    return mid
                elif nums[left]<=target<nums[mid]:
                    right=mid-1
                else:
                    left=mid+1
            elif nums[mid]<=nums[left]:
                if target==nums[mid]:
                    return mid
                elif nums[mid]<target<=nums[right]:
                    left=mid+1
                else:
                    right=mid-1
        return mid if nums[mid]==target else -1

```

# 六十七题
<img width="1030" height="591" alt="image" src="https://github.com/user-attachments/assets/858cff0d-6410-47d8-9926-25e67d4fd417" />
同样的部分单调二分法 思路一样 类比一下
首先left和mid比 如果mid大于left 说明左边是单调 mi给left  
如果left比mid大 说明右边单调 则把mi给mid 然后观察左边
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left,right=0,len(nums)-1
        mid=(left+right)//2
        mi=float('inf')
        while left<=right:
            mid=(left+right)//2
            if nums[mid]>=nums[left]:
                mi=min(mi,nums[left])
                left=mid+1
            elif nums[mid]<=nums[left]:
                mi=min(mi,nums[mid])
                right=mid-1
        return mi

```

# 七十题
<img width="1045" height="1125" alt="image" src="https://github.com/user-attachments/assets/971d3b5c-9035-4bf5-9a99-b7f51ea1766e" />
很有趣的一题 用数组模拟栈 但有需要o（1）复杂度获得最小值 因此我们自然而然的思路是 从一开始输入第一个值就一直维护当前最小栈值 并用元组存入  
**你可以在担心 如果出栈了某个元素 那我之前维护的值不会丢失吗**

这就是栈的性质了 后进先出 因此pop掉的刚好是最新维护的值 pop出去之后 就是次维护的值 刚好是剩下的区间的最小值

**很奇妙的一题** 看起来只是简单的加入了维护 但实际上巧妙的利用了栈的性质
```python
class MinStack:

    def __init__(self):
        self.nums=[(0,inf)]

    def push(self, val: int) -> None:
        self.nums.append((val,min(self.nums[-1][1],val)))
        

    def pop(self) -> None:
        temp=self.nums.pop()

    def top(self) -> int:
        return self.nums[-1][0]

    def getMin(self) -> int:
        return self.nums[-1][1]
```

# 第七十一题
<img width="1038" height="516" alt="image" src="https://github.com/user-attachments/assets/7b4c8556-3374-470a-b48b-6c0cbf145938" />
栈的题目 但是很明显用递归更好解决

一开始使用递推 但是遇到嵌套列表就无法解决  
然后使用递归一开始想的是 遇到列表就往里面传 但是太麻烦了 最后看了题解的想法是 通过切片字符串进行递归 然后每次遇到列表就把大的列表放入递归 然后同时递归列表后面的内容（优先嵌套 然后才是接下去）
```py
self.decodeString(s[i+1:j])*int(s[:i])+self.decodeString(s[j+1:])
```
写成这样

```python
class Solution:
    def decodeString(self, s: str) -> str:
        if not s:
            return s
        if s[0].isalpha():
            return s[0]+self.decodeString(s[1:])
        i=s.find('[')
        balance=1
        for j in count(i+1):
            if s[j]=='[':
                balance+=1
            elif s[j]=="]":
                balance-=1
                if balance==0:
                    return self.decodeString(s[i+1:j])*int(s[:i])+self.decodeString(s[j+1:])
```

# 第七十二题
<img width="1049" height="952" alt="image" src="https://github.com/user-attachments/assets/98d9793e-db41-4265-ba90-919033b3d0b6" />

非常基础的一道单调栈（没想到第一次写单调栈居然不是在力扣而是在课堂hhh没绷住）

以及有经验了 写起来当然得心应手

思路就是遇到大于栈顶元素 就放入ans 然后不断pop 直到小于 再放入栈即可
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        l=len(temperatures)
        stack=[]
        ans=[0 for i in range(l)]
        for i,v in enumerate(temperatures):
            while stack and (v>stack[-1][0]):
                ans[stack[-1][1]]=i-stack[-1][1]
                stack.pop()
            stack.append((v,i))
        return ans  
```

# 第七十三题

<img width="1008" height="915" alt="image" src="https://github.com/user-attachments/assets/2977bc1d-a156-4015-9ff7-44706b947532" />

依旧单调栈 甚至难度小于之前学校给的题目（学校里面要求区间和 还需要加一个前缀和优化）

思路 与其遍历每个区间 不如固定一个值 找到最大区间 通过单调栈 完成对left和right的标记 然后遍历每个值 乘以其最大区间长度即可

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        l=len(heights)
        stack=[]
        left=[0 for i in range(l)]
        right=[l-1 for i in range(l)]
        for i in range(l):
            while stack and (heights[i]<stack[-1][0]):
                right[stack[-1][1]]=i-1
                stack.pop()
            stack.append((heights[i],i))
        stack=[]
        for i in range(l-1,-1,-1):
            while stack and (heights[i]<stack[-1][0]):
                left[stack[-1][1]]=i+1
                stack.pop()
            stack.append((heights[i],i))
        ans=-inf
        for i,v in enumerate(heights):
            ans=max(ans,v*(right[i]-left[i]+1))
        print(left,right)    
            
        return ans
        
```
# 第74题
<img width="965" height="652" alt="image" src="https://github.com/user-attachments/assets/e4553267-ca57-4d51-acfe-77f51e3a0b7c" />

数组中选第k小的值 需要O（n）复杂度

题解的做法是快速排序选择 但是我看到评论区有用桶排序的做法 感觉很厉害
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        ma=max(nums)
        mi=min(nums)
        has=defaultdict(int)
        count=0
        for i in nums:
            has[i]+=1
        for i in range(ma,mi-1,-1):
            count+=has[i]
            if count>=k:
                return i
```
原理是通过构造一个default字典 然后找到最大值和最小值 从大往小遍历 遇到几个就加几个（桶排序适用于计数 因此面对这种求第k个大小的题目很适配）

每次加上has[i] 如果没有这个数就是+0 然后  
if >=k 说明一个桶里面可能有多个数 因此超出了也是放这个数出来

**总结**  
桶排序适用于计数（第k个数 ）而非排序 并且需要最大值不太大的 否则空间消耗过大

# 第75题
<img width="1059" height="816" alt="image" src="https://github.com/user-attachments/assets/908dc7ba-a87c-48b2-88d5-8c020c521c04" />
依旧桶排序 遇到前k个直接硬套 复杂度可以在o（n）

思路是 先统计每个数的次数 然后反过来将字典key当作次数 将value当作出现的数 再利用桶排序从大到小遍历的性质 将值输出

用到一个比较好玩的地方就是extend 直接将列表列入而不是append嵌套 还有使用了defaultdict（list）

**想说的是 我现在对于列表推导式依旧运用的如火纯青啦**
```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        has=defaultdict(int)
        ans=[]
        count=0
        for i in nums:
            has[i]+=1
        arr=[(i,has[i]) for i in has.keys()]
        arr_temp=[i[1] for i in arr]
        print(arr,arr_temp)
        has2=defaultdict(list)
        ma=max(arr_temp)
        mi=min(arr_temp)
        for i,v in arr:
            has2[v].append(i)
        print(has2)
        for i in range(ma,mi-1,-1):
            ans.extend(has2[i])
            count=len(ans)
            if count>=k:
                return ans
```
# 第76题
<img width="1010" height="1072" alt="image" src="https://github.com/user-attachments/assets/9ccdee02-d2ba-47c0-9b80-41101ac2b5f9" />
中位数 做法就是分成两个区间 保证左边小于右边 并且左边永远和右边个数相等或者多一

如何维护区间呢 使用两个堆 一个最小一个最大（没有最大堆 我们在传入时使用负值）

如果相等 那么就先放右 然后把右边的最小值拿出来 放到左边 保证相等

如果不等 就放左 然后把左边最大值拿出来 放右边 保证多一

这样不断维护最值即可
```python
class MedianFinder:

    def __init__(self):
        self.left=[]
        self.right=[]


    def addNum(self, num: int) -> None:
        if len(self.left)==len(self.right):
            heappush(self.left,-heappushpop(self.right,num))
        else:
            heappush(self.right,-heappushpop(self.left,-num))

    def findMedian(self) -> float:
        if len(self.left)>len(self.right):
            return -self.left[0]
        return (self.right[0]-self.left[0])/2


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```
# 第77题
<img width="1047" height="783" alt="image" src="https://github.com/user-attachments/assets/c9c1cd65-be3c-4abc-a7c0-ce4a80e36e5f" />
终于到贪心了 但是实际上跟贪心没啥关系 主要思路是 一次的遍历中不断维护最小值和最大答案
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans=0
        mi=prices[0]
        for i in prices:
            ans=max(ans,i-mi)
            mi=min(mi,i)
        return ans
```
其实我总结了一下 贪心的思想很可能就是 从一个是两个都在动的变量 转化成固定一个值 然后遍历该数组 在这道题的体现就是 遍历的过程中维护最小值 然后每到一个值就减去这个最小值 从而得到答案
# 第78题 
<img width="1068" height="741" alt="image" src="https://github.com/user-attachments/assets/16fedd61-c596-4140-a92b-82a09eb5288f" />
依旧是一个贪心 总结一下 依旧是在单次的遍历中 通过a=max(a,x)来维护一个值 从而实现某些想法（贪心思想）  
这一题的想法是 遍历每一个位置 同时维护能够到达的最远位置 知道最远位置比当前位置小 则不能继续向前

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goto=0
        for i,v in enumerate(nums):
            goto=max(goto,i+v)
            if goto<=i and i!=len(nums)-1:
                return False
        return True
```
代码非常简单

