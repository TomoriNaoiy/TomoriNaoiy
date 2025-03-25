# c++的类
### public和private的区别
private只能在类中访问 也就是在子函数里面可以修改或者调用这个变量
public能在类外访问 也就是在主函数里面也可以直接修改这个对象的某个变量和调用函数
```c++
class MyClass {
private:
    int privateVar; // 私有成员变量
    void privateMethod() { // 私有成员函数
        cout << "woshishiyou"<< endl;
    }
public:
    void setPrivateVar(int value) { // 公共方法访问私有成员
        privateVar = value;//这里可以在类中 给privateVar赋值
    }
    void showPrivateVar() {
        cout << "privateVar: " << privateVar << endl;
    }//这里就不必再传入参数
};

int main() {
    MyClass obj;
    // obj.privateVar = 10; // 错误：无法直接访问私有成员 如果前面的privateVar在public里面 就可以在这里直接修改 在这里只能给子函数传入参数
    // obj.privateMethod(); // 错误：无法直接调用私有方法
    obj.setPrivateVar(10); // 通过公共方法间接访问私有成员
    obj.showPrivateVar(); // 输出：privateVar: 10
    return 0;
}
```

**注意**正常的类的使用 假如不直接给类传入参数 那么正常的写法就是
```c++
#include <iostream>
using namespace std;
class node {
public:
    int num;
    int sum(int value)
    {
        return value * 2;
    }
};
int main()
{
    int num;
    cin >> num;
    node node1;
    cout << node1.sum(num);//在调用函数时给参数 这样就用不上类里面的num变量了 但我们一般不这么用
}
一般的做法 类似python的想法 在创建对象的时候就把参数传入 然后在类里面的子函数对这些参数进行变化 见下
```c++
#include <iostream>
using namespace std;
class node {
public:
    int num;
    node(int N) :num(N) {}//初始化
    //也可以写成 node(int N){num=N;}//注意 要写分号不是逗号
    int sum()//无需参数
    {
        return num * 2;
    }
};
int main()
{
    int num;
    cin >> num;
    node node1(num);
    cout << node1.sum();
}
```
几个注意点
1. 要在类里面初始化函数 node（int N ）：num（N）{}
2. 注意 初始化时 冒号后面是类里面的变量（零时变量）＋{} 这段话的意思是 将传入node参数的N 作为num
3. 成员函数 不需要再给参数 直接使用类里面的数
4. 调用时 创建对象时要给参数 调用函数用。 并且函数要加（）但不给参数
## 一些奇巧淫技
在类里面定义函数 在外面构造
```c++
class stdu
{
public:
    void print();
};
void stdu::print()
{
balabal
}
```
### 析构函数 在删除的时候调用 并且如果不写delete的话 后建立的会先删除 也就是先调用苟晞函数

```c++
class Student{
    int rank;
    string name;
    public:
        int getRank(){return rank;    }
        Student(string name, int rank):name(name), rank(rank){    }
        ~Student(){ cout<<name<<endl;}
};
Student a(1,"aaa")
delete a;
```
当然这里面还蕴藏了类的数组的构造
```
Student *pS[SIZE]
 while(count<SIZE && rank>0){
        cin>>name;
        pS[count++]= new Student(name, rank);
```
### 类中的静态变量 
```c++
class aa
{
public:
stastic int c;
int a
aa(int a):a(a){c++}
}；
int aa::c=0;
int main()
{
cout<<aa::c;
}
```
这里就是静态变量 可以用作计数 每次创建对象的时候给c+1 用作计数 后面调用的时候要aa：：c 并且要声明
### const
常变量的设置

在类里面
```
class a
{
int plus(int a) const
{
balaba;
}
int plus(int a) 
{
balaba;
}
}
//然后调用的时候 使用const定义 就会只调用对应类里面的常成员函数
const a hahah；
a ahaaha;
hahah.plus();
ahaaha.plus();
```
### this
这个很好理解 就是pyhton里面的self
```c++
class a
{
public:
int my;
void set(int my)
{
this->my=my
}
};

// 返回当前对象
class Student {
    int age;
public:
    Student& setAge(int age) {
        this->age = age;
        return *this; // 返回当前对象
    }
};
//访问成员函数
class Student {
public:
    void print() {
        cout << "Student" << endl;
    }
    void display() {
        this->print(); // 调用当前对象的成员函数
    }
};
```







**~~好啦 你已经会使用类了 快去写链表把~~**
# 链表
话不多说 直接开始链表
```c++
#include<iostream>
using namespace std;
class node
{
    public:
    int val;
    node* next;//创建指针
    node (int val):val(val),next(NULL){}//初始化
};
//这里是创建节点类 在后续的链表中 每个节点（val和next指针）都是一个类 
class link
{
public:
    node* head;//每个链表都有一个头节点 一般默认为空
    link():head(NULL){}
    void add(int val)//将数值加到末尾
    {
        node* newnode=new node(val);//先创建一个新节点（开辟空间）
        if (! head)//先判断是否有头（链表是否空） 若空则先给头 
        {
            head=newnode;
            return;
        }
        else//若非空 从头节点开始 一个个向后推 直到最后一位 将末节点的next给newnode
        {
            node* curnode=head;
            while (curnode->next)
            {
                curnode=curnode->next;
            }
        curnode->next=newnode;
        }

    }
    void del(int val)//删除节点
    {
        node* curnode=head;//从头开始
        while(curnode->next)
        {
            if(curnode->next->val==val)
            {
                curnode->next=curnode->next->next;//如果下一格节点要删除 就把当前节点的next指向下一的下一个 这样中间就没了
                return;
            }
            curnode=curnode->next;
        }
    }
    void print()
    {
        node* curnode=head;
        while(curnode)
        {
            cout<<curnode.val<<" ";
            curnode=curnode->next;
        }
    }

};
int main()
{
//qwq
}
```
以上就是基本的链表 下面将一下c++自带的list库
## list
```c++
#include<iostream>
#include<list>
using namespace std;
int main()
{
    list<int> list1;//创建名为list1 int类型的双向链表
    for(int i=0;i<n;i++>)
    {
        int num;
        cin>>num;
        list1.push_back(num);
    }
    for(auto it=list1.begin();it!=list1.end();++it)
    {
        cout<<*it;//it是迭代器 这里获得值需要*
    }
}
```
以上就是list最基本的使用 以下列举一些list链表的用法
1. lst1.push_back(value); // 在链表尾部插入元素
2. lst1.push_front(value); // 在链表头部插入元素
3. lst1.pop_back(); // 移除链表尾部元素
4. lst1.pop_front(); // 移除链表头部元素
5. lst1.insert(iterator, value); // 在指定位置插入元素
6. lst1.erase(iterator); // 移除指定位置的元素


访问元素：

1. lst1.front(); // 获取链表头部元素的引用
2. lst1.back(); // 获取链表尾部元素的引用


遍历列表：

1. for (auto it = lst1.begin(); it!= lst1.end(); ++it) // 使用迭代器访问元素
2. for (const auto& element : lst1) // 使用范围-based for 循环


获取列表大小和判空：

1. lst1.size(); // 获取列表中元素的数量
2. lst1.empty(); // 检查列表是否为空


其他操作：

1. lst1.clear(); // 清空列表
2. lst1.reverse(); // 反转列表
3. lst1.sort(); // 对列表进行排序s
