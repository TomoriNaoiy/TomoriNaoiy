

在sosd的二轮面试中 开始进行对git的学习 此前主要学习对git的各种方法的使用 这边通过[note](https://missing-semester-cn.github.io/2020/version-control/)进行理论学习

**由于对git使用已经较为熟练 这边就跳过练习部分 记录一些概念性的东西以及一些key**
# git的提交是一个有向无环图
<img width="1071" height="186" alt="image" src="https://github.com/user-attachments/assets/e15a8f65-9f3e-4f2a-a18a-79e8848d1690" />
其中 每个o就是一次提交

<img width="993" height="262" alt="image" src="https://github.com/user-attachments/assets/7cedbc2b-c35c-4562-8245-2db18c28cdc0" />
在这里 第三次提交就形成了分叉 这是工作需要的分工合作 那么在最后可以进行‘合并’

# add . 暂存区

这个区域有什么用呢？

例如，考虑如下场景，您开发了两个独立的特性，然后您希望创建两个独立的提交，其中第一个提交仅包含第一个特性，而第二个提交仅包含第二个特性。或者，假设您在调试代码时添加了很多打印语句，然后您仅仅希望提交和修复 bug 相关的代码而丢弃所有的打印语句。

# 概念差不多就这些 接下来记录一下一些用法

- git help <command>: 获取 git 命令的帮助信息
- git init: 创建一个新的 git 仓库，其数据会存放在一个名为 .git 的目录下
- git status: 显示当前的仓库状态
- git add <filename>: 添加文件到暂存区
- git commit: 创建一个新的提交
- git log: 显示历史日志
- git log --all --graph --decorate: 可视化历史记录（有向无环图）
- git diff <filename>: 显示与暂存区文件的差异
- git diff <revision> <filename>: 显示某个文件两个版本之间的差异
- git checkout <revision>: 更新 HEAD（如果是检出分支则同时更新当前分支）
- git remote: 列出远端
- git remote add <name> <url>: 添加一个远端
- git push <remote> <local branch>:<remote branch>: 将对象传送至远端并更新远端引用(git push origin main)
- git branch --set-upstream-to=<remote>/<remote branch>: 创建本地和远端分支的关联关系
- git fetch: 从远端获取对象/索引
- git pull: 相当于 git fetch; git merge
- git clone: 从远端下载仓库
- git remote -v 查看一下push的地址
# 一些经常遇到的问题

- push的文件记得save 不然上传的就是没改的！
- fatal: unable to access 'https://github.com/TomoriNaoiy/collection-ai/': Failed to connect to github.com port 443 after 21060 ms: Could not connect to server这个报错是网络问题 开个加速器就行



