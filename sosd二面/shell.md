# shell学习
首先 第一部也是最重要的一部 在wsl上面写脚本
```
下载完wsl 想要运行
wsl
```
发现
<img width="670" height="71" alt="image" src="https://github.com/user-attachments/assets/158ee700-8a8f-4171-b0c5-0ed554a2dc2c" />
了解了一下知道 wsl只是一个兼容层 能够让你在win上面使用Linux 那么实际对于的版本需要我们自己下载 比如我下载了ubuntu

### 那么正式开始
首先 跟大部分语言一样 都有复制 if while等等操作 但是有几个点 
1. 不可以```a = c``` 只能a=c 等号有特殊含义
2. ‘’是变量 “”才是字符串 比如echo（'a'） echo("a")
3. 同样的函数 但是不用分号
```
md(){
mkdir -p "$1"
    cd "$1"
}
```
里面的$是脚本参数
- $0 - 脚本名
- $1 到 $9 - 脚本的参数。 $1 是第一个参数，依此类推。
- $@ - 所有参数
- $# - 参数个数
- $? - 前一个命令的返回值
- $$ - 当前脚本的进程识别码
- !! - 完整的上一条命令，包括参数。常见应用：当你因为权限不足执行命令失败时，可以使用 sudo !! 再尝试一次。
- $_ - 上一条命令的最后一个参数。如果你正在使用的是交互式 shell，你可以通过按下 Esc 之后键入 . 来获取这个值。

- 还有就是一些操作 但是具体有什么作用 还不太清楚
```
false || echo "Oops, fail"
# Oops, fail

true || echo "Will not be printed"
#

true && echo "Things went well"
# Things went well

false && echo "Will not be printed"
#

false ; echo "This will always run"
# This will always run
```

### 命令替换
当您通过 $( CMD ) 这样的方式来执行 CMD 这个命令时，它的输出结果会替换掉 $( CMD ) 。例如，如果执行 for file in $(ls) ，shell 首先将调用 ls ，然后遍历得到的这些返回值。还有一个冷门的类似特性是 进程替换（process substitution）， <( CMD ) 会执行 CMD 并将结果输出到一个临时文件中，并将 <( CMD ) 替换成临时文件名。这在我们希望返回值通过文件而不是 STDIN 传递时很有用。例如， diff <(ls foo) <(ls bar) 会显示文件夹 foo 和 bar 中文件的区别。
<img width="874" height="519" alt="image" src="https://github.com/user-attachments/assets/4d0171ef-4f0d-47d9-afa6-32dca7bc8fa0" />

# 通配符 
跟正则表达式一模一样 
？匹配单个字符 *匹配多个字符
rm roo？ rm roo* 删掉rool roo1 roo3 后者删掉所有roo----有关的

# 查找文件

程序员们面对的最常见的重复任务就是查找文件或目录。所有的类 UNIX 系统都包含一个名为 find 的工具，它是 shell 上用于查找文件的绝佳工具。find 命令会递归地搜索符合条件的文件，例如：
```
# 查找所有名称为src的文件夹
find . -name src -type d
# 查找所有文件夹路径中包含test的python文件
find . -path '*/test/*.py' -type f
# 查找前一天修改的所有文件
find . -mtime -1
# 查找所有大小在500k至10M的tar.gz文件
find . -size +500k -size -10M -name '*.tar.gz'
```

# 查找代码
```
# 查找所有使用了 requests 库的文件
rg -t py 'import requests'
# 查找所有没有写 shebang 的文件（包含隐藏文件）
rg -u --files-without-match "^#\!"
# 查找所有的foo字符串，并打印其之后的5行
rg foo -A 5
# 打印匹配的统计信息（匹配的行和文件的数量）
rg --stats PATTERN
```

# 练习作业
### 作业1
<img width="1060" height="577" alt="image" src="https://github.com/user-attachments/assets/7c651eba-c2d9-4f35-8d9c-c652dc02ca7c" />

ls方法
```
ls -alh --color=auto -lt # ls列出文件 
```

-a：显示所有文件和目录，包括隐藏的（以.开头的）。

-l：使用长格式列出信息，包括文件权限、所有者、大小和最后修改时间。

-h：以人类可读的格式显示文件大小（例如，1K、234M、2G）。

--color=auto：自动启用颜色输出，以便更清晰地区分不同类型的文件。

-lt：按修改时间排序，最新的文件显示在最前面。
<img width="1720" height="672" alt="image" src="https://github.com/user-attachments/assets/be98c318-b1ca-4c2f-b4cd-a63eb5c3a8bc" />
效果如上

# 作业2
编写两个 bash 函数 marco 和 polo 执行下面的操作。 每当你执行 marco 时，当前的工作目录应当以某种形式保存，当执行 polo 时，无论现在处在什么目录下，都应当 cd 回到当时执行 marco 的目录。 为了方便 debug，你可以把代码写在单独的文件 marco.sh 中，并通过 source marco.sh 命令，（重新）加载函数。
```
#!/bin/bash
 marco() {
     export MARCO=$(pwd)
 }
 polo() {
     cd "$MARCO" # 返回
 }
```

# 作业3

假设您有一个命令，它很少出错。因此为了在出错时能够对其进行调试，需要花费大量的时间重现错误并捕获输出。 编写一段 bash 脚本，运行如下的脚本直到它出错，将它的标准输出和标准错误流记录到文件，并在最后输出所有内容。 加分项：报告脚本在失败前共运行了多少次。
```
#!/usr/bin/env bash

 n=$(( RANDOM % 100 ))

 if [[ n -eq 42 ]]; then
     echo "Something went wrong"
     >&2 echo "The error was using magic numbers"
     exit 1
 fi

 echo "Everything went according to plan"
```
# 作业4

本节课我们讲解的 find 命令中的 -exec 参数非常强大，它可以对我们查找的文件进行操作。但是，如果我们要对所有文件进行操作呢？例如创建一个 zip 压缩文件？我们已经知道，命令行可以从参数或标准输入接受输入。在用管道连接命令时，我们将标准输出和标准输入连接起来，但是有些命令，例如 tar 则需要从参数接受输入。这里我们可以使用 xargs 命令，它可以使用标准输入中的内容作为参数。 例如 ls | xargs rm 会删除当前目录中的所有文件。

您的任务是编写一个命令，它可以递归地查找文件夹中所有的 HTML 文件，并将它们压缩成 zip 文件。注意，即使文件名中包含空格，您的命令也应该能够正确执行（提示：查看 xargs 的参数 -d，译注：MacOS 上的 xargs 没有 -d，查看这个 issue）

如果您使用的是 MacOS，请注意默认的 BSD find 与 GNU coreutils 中的是不一样的。你可以为 find 添加 -print0 选项，并为 xargs 添加 -0 选项。作为 Mac 用户，您需要注意 mac 系统自带的命令行工具和 GNU 中对应的工具是有区别的；如果你想使用 GNU 版本的工具，也可以使用 brew 来安装。

```
find . -type f -name "*.html" | xargs -d '\n'  tar -cvzf html.zip
```
很好理解 在cs50里面也经常遇到args的命令行传参 这里输出参数后进行搜索压缩

# 作业5
编写一个命令或脚本递归的查找文件夹中最近修改的文件。更通用的做法，你可以按照最近的修改时间列出文件吗？
```
find . -type f -print0 | xargs -0 ls -lt | head -1

# 按修改时间 只需要通过 head -1即可
```
<img width="1340" height="56" alt="image" src="https://github.com/user-attachments/assets/a3817de7-3465-46f0-9e2e-49be1bd2e63f" />

