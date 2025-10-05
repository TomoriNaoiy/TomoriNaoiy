n = int(input())
inl = list(map(int, list(input().strip())))
oul = list(map(int, list(input().strip())))
stack = []
ops = []
cur = 0
flag = True
for target in oul:
    while (not stack or stack[-1] != target) and cur < n:
        stack.append(inl[cur])
        ops.append("in")
        cur += 1
    if stack and stack[-1] == target:
        stack.pop()
        ops.append("out")
    else:
        flag = False
        break
if flag:
    print("Yes")
    for op in ops:
        print(op)
else:
    print("No")
