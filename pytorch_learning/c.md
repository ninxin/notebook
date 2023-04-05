 ```c
%d - 打印10进制整形
%c - 打印字符
%f - 打印浮点数 - float
%lf - 打印双精度浮点数 - double
%p - 以地址形式打印
%x - 打印16进制数字
 ```



```c
printf("%d\n", sizeof(char));		// 1字节	8 比特 = 1 字节	比特存 0 / 1
printf("%d\n", sizeof(short));		// 2
printf("%d\n", sizeof(int));		// 4
printf("%d\n", sizeof(long));		// 4/8		long >= int		编译器不同
printf("%d\n", sizeof(long long));		// 8
printf("%d\n", sizeof(float));		// 4
printf("%d\n", sizeof(double));		// 8
```



局部变量的作用域是变量所在的局部范围

全局变量的作用域是整个工程（当在当前工程不同文件，可用extern声明外部符号）



##### 常量

字面常量：如数字

const修饰的常变量：是变量，但又有常属性

#define定义的标识符常量：这就可以定义数组时使用

枚举常量：enum

```c
enum Sex {
	male,			// 0
	female,		// 1
	secret		  // 2
};
```



```c
// "abc" -- 'a', 'b', 'c', '\0' -- '\0' 转义字符，字符串的结束标志
// '\0' -- 0
// 'A' -- 97	ASCII表
char arr1[] = "abc";
char arr2[] = { 'a', 'b', 'c' };
printf("%s\n", arr1);			// abc
printf("%s\n", arr2);			// abc烫烫烫烫烫烫烫烫烫烫烫烫烫烫烫烫虜?~?
printf("%d\n", strlen(arr1));		// 3
printf("%d\n", strlen(arr2));		// 大于3的值
```



```c
char arr1[] = "abc";
char arr2[] = { 'a', 'b', 'c' , 0};
printf("%s\n", arr1);			// abc
printf("%s\n", arr2);			// abc
```



```c
// \ddd ddd表示1~3个八进制数字
// \xdd dd表示2个十六进制数字

// \32作为8进制代表的10进制数字，作为ASCII码值对应的字符
printf("%c\n", '\32')
printf("%d\n", strlen("c:\test\32\test.c"));		// 13 \t,\32,\t转义字符，算一个字符
```



```c
// 移位运算符： << 左移  >> 右移, 变成二进制再移动
int a = 1;			// 0...01  共32位
int b = a << 2;		// 0...0100 全部向左移2位，多的消失，少的补零		a本身不变
printf("%d\n", b);		// 4

// (2进制)位操作符：
// &与 |或 ^异或（相同为0，相异为1）
int a = 3;			// ...011
int b = 5;			// ...101

int c = a & b;		// ...001		1
int d = a | b;		// ...111		7
int e = a ^ b;		// ...110		6
printf("%d\n", c);
printf("%d\n", d);
printf("%d\n", e);
```



```c
// 单目操作符
!(逻辑反操作) -(负号) +(正号，一般省略) &(取地址) sizeof(算变量或类型所占的字节)
~(二进制按位取反)	--	++	(类型)如：(int)3.14 强制类型转换
// 双目操作符

// 三目操作符
```



只要是整数在内存中存储的都是补码

正数三码相同

```c
// ~ 按二进制位取反
// 00000000 00000000 00000000 00000000
int a = 0;		// 有符号整型，首位代表正负，0代表正，1代表负
// 11111111 11111111 11111111 11111111	补码
// 11111111 11111111 11111111 11111110	反码
// 10000000 00000000 00000000 00000001	原码	-1
int b = ~a;

// 原码， 反码， 补码
// 原码符号位不变， 其他位按位取反， 反码加1得到补码
// 补码减1得到反码，反码符号位不变，其他位按位取反得到原码

// 负数在内存中存储的是二进制的补码
printf("%d\n", b);		// 打印的是这个数的原码	
```



###### 逻辑运算符

```CQL
// && 逻辑与
int a = 3;
int b = 5;
int c = a && b;		// 1
// || 逻辑或
```



###### 条件操作符/三目操作符

```c
// exp1 ? exp2 : exp3		如果exp1成立则是exp2，否则是exp3
int a = 1;
int b = 2;
int c = (a > b ? a : b)
```



```c
int a = 10;
int b = a++;			// 后置的++， 先使用，再++， 也就是先用a对b赋值， 在对a++
printf("%d\n", a);		// 11
printf("%d\n", b);		// 10

int c = ++a;			// 前置++， 先对a++，再对c赋值
printf("%d\n", a);		// 12
printf("%d\n", c);		// 12
```



###### 关键字

```c
// typedef	重新定义类型
typedef unsigned int uint;	// 将unsigned int重命名，以后可以通过使用uint来调用
// static	修饰局部变量，变成静态的局部变量，不像局部变量用完即销毁，可能会继续保留，延长生命周期
// 用static修饰全局变量时，使用extern引入外部符号就不行了，改变了全局变量的作用域，让他只能在它的源文件使用
// 也可修饰函数，和全局变量类似。普通的函数可被外部引用，修饰后只能在内部使用。

```



### 指针（一个存放地址的变量）

```c
int a = 10;		// 开辟一块空间，名字为a，存放10
// 指针的类型和指向地址元素的类型一样，即int-int， char-char。。。
int* p = &a;	// 取得a的地址，赋给指针p.	开辟另一块空间p存放a的地址
*p = 20;		// 解引用， 将地址a存放的数变为20。	*p相当于a
```

指针大小在32位平台是4个字节，在64位平台是8个字节



### 结构体

```c
// 创建一个结构体类型
struct Book
{
	char name[20];
	short price;
};

int main(){
    // 利用结构体类型-创建一个该类型的变量
    struct Book b1 = { "C语言程序设计", 55 };
    
    struct Book* pd = &b1;
    // . 结构体变量.成员
    // -> 结构体指针->成员
	printf("%s\n", (*pd).name);
	printf("%d\n", (*pd).price);    
    
	printf("%s\n", pd->name);
	printf("%d\n", pd->price);    
    
	printf("书名:%s\n", b1.name);
	printf("价格:%d\n", b1.price);
    
    // price是变量，可以直接赋值修改
    b1.price = 15;
	printf("修改后的价格:%d\n", b1.price);
    // name是数组，不能直接赋值修改
    strcpy(b1.name, "C++");		// 对字符串拷贝
	printf("修改后的书名:%s\n", b1.name);
    
	return 0;
}
```

### 注意：

```c
int a = 10;
20 < a && a < 30
if (20 < a < 30)		//不能这么写，相当于(20<10为0)--(0<30)成立
    ...
```

**else和离得最近的未匹配的if相匹配**

```c
switch(整型表达式){
    case 整型常量表达式:
        语句;
        break;
    ...
}
```

```c
int day = 3;
// 更简洁
switch(day){
    case 1:		// 没有break则会继续执行下去
    case 2:
    case 3:
    case 4:
    case 5:
        printf("工作日\n");
        break;
    case 6:
    case 7:
        printf("休息日\n")
            break;
    default:		// 可以放前面，没有顺序可言
        printf("输入错误\n");
        break;
}
```



```c
// 当接受的是数字时，返回
int main(){
    int ch = 0;
    // EOF为文件结束标志 -> -1
    while((ch = getchar()) != EOF){		// 如果接受的字符为ctrl+z则不执行
        if (ch < '0' || ch > '9')
        	continue;
        putchar(ch);		// 输出所得的字符
    }
    return 0;
}
```



```c
// for省略初始化， for的初始化；判断；调整 都可以省略
int i = 0;
int j = 0;
for (;i < 10; i++){
    // 第一次进去循环i=0，然后j一直增加到10，退出第二个循环，然后i=1，但是j还是10，不执行第二个循环
    for (; j < 10; j++){
        printf("hh\n");
    }
}		// 输出10个hh
```



```c
// 下列循环几次
int i = 0;
int j = 0;
// 判断j=0这个表达式为真还是假，0代表假，不进入循环
for (i = 0, j = 0; j = 0; i++, j++){
    j++
}
// 0次
```

##### 计算阶乘

```c
int jiecheng(int x) {
	if (x > 1)
		x *= jiecheng(x - 1);
	else
		return x;
	
}
int main(){
    int c = 10;
	printf("%d\n", jiecheng(c));

	return 0;
}
```

##### 计算1-10阶乘的和

```c
int i = 0;
int n = 0;
int ret = 1;
int sum = 0;
for (n = 1; n < 11; n++){
    ret *= n;
    sum += ret;
}
printf("d\n", sum);
```

##### 二分法查找有序数组

```c
int arr[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
int k = 7;	
int left = 0;
int sz = sizeof(arr) / sizeof(arr[0]);	// 得到数组长度
int right = sz - 1;

while (left <= right) {		// 当left > right结束循环
    int mid = (left + right) / 2;
    if (arr[mid] < k) {
        left = mid + 1;
    }
    else if (arr[mid] > k) {
        right = mid - 1;
    }
    else {
        printf("找到了，下标是：%d\n", mid);
        break;
    }
}
if (left > right) {
    printf("找不到\n");
}
```

##### 将字符串从无到有逐行从两端打印

```c
###################
w#################!
we###############!!
wel#############!!!
welc###########!!!!
welco#########!!!!!
welcom#######t!!!!!
welcome#####it!!!!!
welcome ###bit!!!!!
welcome t# bit!!!!!

	// 和整形数组不同，字符串最后还有个\0
    // 用两个数组，一个接受字符串，另一个接受变化的字符串
	char arr1[] = "welcome to bit!!!!!";
	char arr2[] = "###################";
	int left = 0;
	//int right = sizeof(arr1) / sizeof(arr1[0]) - 2;		// 得到最后一个元素下标，因为最后哈有\0，所以要-2
	int right = strlen(arr1) - 1;
	while (left <= right) {
		printf("%s\n", arr2);
		arr2[left] = arr1[left];
		arr2[right] = arr1[right];
		Sleep(1000);		// 休息1000毫秒再执行下一步
		system("cls");		// 执行系统命令。 cls -- 清空屏幕		
		left++;
		right--;
	}
```

##### == 不能比较两个字符串是否相等，应该用库函数--strcmp



##### 写函数来交换数字

```CQL
void Swap(int* pa, int* pb) {
	int tmp = 0;
	tmp = *pa;
	*pa = *pb;
	*pb = tmp;
}


	int a = 10;
	int b = 20;
	Swap(&a, &b);
	printf("a=%d, b=%d\n", a, b);
```

###### 当实参传给形参时，形参是对实参的一份临时拷贝，对形参的修改不会改变实参



**数组传到函数传的是首元素的地址，所以不能在函数内部计算数组的长度**

#### 函数调用

##### 传值调用：

函数的形参和实参分别占用不同的内存块，对形参的修改不会影响实参

##### 传址调用：

把函数外部创建的变量的内存地址传给函数参数的一种调用函数的方式

可以让函数和函数外边的变量建立真正的联系，也就是函数内部可以直接操作函数外部的变量



##### 得到100-200之间的素数

```c
// 判断是否为素数，是则返回1，否则返回0
int is_prime(int n) {
	int i = 0;
	for (i = 2; i < n / 2; i++) {
		if (n % i == 0) {
			return 0;
		}
	}
	return 1;
}
int main() {
	int i = 0;
	for (i = 100; i <= 200; i++) {
		if (is_prime(i)==1) {
			printf("%d是素数。\n", i);
		}
	}
}
```

##### 得到1000-2000的闰年

```c
// 判断是否为闰年
int is_leap_year(int year) {
	if ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)) {
		return 1;
	}
	return 0;
}

int main() {
	int year = 0;
	for (year = 1000; year < 2001; year++) {
		if (is_leap_year(year) == 1) {
			printf("%d\n", year);
		}
	}

	return 0;
}
```

**++优先级大于\*，所以要（\*p）++，而不是*p++**



```c
// printf返回的是打印字符的个数
int main(){
    printf("%d", printf("%d", printf("%d", 43)));		//4321
    // 相当于
    //printf("%d", printf("%d", 2));
    //printf("%d", 1)
}
```

##### 函数声明，函数名(...);	当函数在main后需要声明才不会报错



栈区：局部变量，函数形参

堆区：动态开辟的内存，malloc，calloc

静态区：全局变量，static修饰的变量



##### 递归--可能栈溢出

将数字从左到右逐个输出

```c
void print(int n) {
	if (n > 9) {		// n不是个位数则一直递归，去掉尾数，直到是个位数，然后输出个位数。	从而达到将数字从左到右输出
		print(n / 10);
	}
	printf("%d ", n % 10);
}

int main() {
	unsigned int num = 0;
	scanf("%d", &num);
	print(num);
}
```

```c
// 不用创建临时变量计算字符串长度
int my_strlen(char* str) {
	int count = 0;
	while (*str != '\0') {
		count++;
		str++;
	}
	return count;
}
int main() {
	char arr[] = "bit";
	int len = my_strlen(&arr);
	printf("%d\n", len);
	return 0;
}
```

##### 递归的方法计算字符串的长度

```c
// 当字符不是‘\0’，指针指向下一个，长度加一
int my_strlen(char* str) {
	if (*str != '\0') {
		return 1 + my_strlen(str+1);
	}
	return 0;
}

int main() {
	char arr[] = "bit";
	int len = my_strlen(&arr);
	printf("%d\n", len);
	return 0;
}
```

##### 递归求阶乘

```c
int Facl(int n) {
	if (n > 1) {
		return n * Facl(n - 1);
	}
	return 1;
}
int main() {
	// n的阶乘
	int n = 0;
	int ret = 0;
	scanf("%d", &n);
	ret = Facl(n);
	printf("%d", ret);
	return 0;
}
```

##### 递归求斐波那契数

```c
int Fib(int n) {
	if (n<=2) {
		return 1;
	}
	return Fib(n - 1) + Fib(n - 2);		// 效率低，重复计算多
}
int main() {
	int n = 0;
	int ret = 0;
	scanf("%d", &n);
	ret = Fib(n);
	printf("%d\n", ret);
	return 0;
}
```

##### 循环方式--更快

```c
int Fib(int n) {
	int a = 1;		// n=1的斐波那契数
	int b = 1;		// n=2
	int c = 1;		// 用来存储前两个数之和

	while (n > 2) {
		c = a + b;
		a = b;
		b = c;
		n--;
	}
	return c;
}
int main() {
	int n = 0;
	int ret = 0;
	scanf("%d", &n);
	ret = Fib(n);
	printf("%d\n", ret);
	return 0;
}
```



```c
// strlen 只用用于字符串	库函数--要引用头文件
// sizeof 计算变量、数组、类型的大小 -- 单位是字节	操作符
int main() {
	char arr[] = "abcdef";
	printf("%d\n", sizeof(arr));	// 7 计算所占空间大小，字符串后面还有\0,占一个字节
	printf("%d\n", strlen(arr));	// 6 计算\0前的字符数
	return 0;
}
```

```c
int main() {
	char arr1[] = "abc";
	char arr2[] = { 'a', 'b', 'c' };
	printf("%d\n", sizeof(arr1));	// 4 结尾存储了\0
	printf("%d\n", sizeof(arr2));	// 3 结尾没有\0
    printf("%d\n", strlen(arr1));	// 3
	printf("%d\n", strlen(arr2));	// 大于3的随即数，因为没有\0
	return 0;
}
```

##### 冒泡排序

```c
void bubble_sort(int arr[], int sz) {
	// 先确定冒泡排序的趟数，n-1
	int i = 0;
	for (i = 0; i < sz - 1; i++) {
		int flag = 1;		// 假设现在是有序的
		// 每一趟排序
		int j = 0;
		for (j = 0; j < sz - 1 - i; j++) {
			if (arr[j] > arr[j + 1]) {
				int tmp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = tmp;
				flag = 0;		// 交换过就说明不是有序的，打破了假设，变为0
			}
		}
		if (flag == 1) {
			break;		// 如果真是有序的，则不用继续了，直接退出
		}
	}
}


int main() {
	int arr[] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
	int i = 0;
	int sz = sizeof(arr) / sizeof(arr[0]);
	bubble_sort(arr, sz);
	for (i = 0; i < sz; i++) {
		printf("%d ", arr[i]);
	}
	return 0;
}
```

**数组名代表的是数组首元素的地址，两种情况除外**：

1.sizeof(数组名)-表示整个数组，计算的是整个数组的大小，单位是字节

2.&数组名，表示整个数组，取出整个数组的地址

```c
int main() {
	int arr[] = { 1, 2, 3, 4 };
	printf("%p\n", arr);		// 0000006325CFF748
	printf("%p\n", &arr[0]);	// 0000006325CFF748		
	printf("%p\n", &arr);		// 0000006325CFF748

	printf("%p\n", arr+1);		// 0000006325CFF74C
	printf("%p\n", &arr[0]+1);	// 0000006325CFF74C
	printf("%p\n", &arr+1);		// 0000006325CFF758
	return 0;
}
```

#### 算术操作符

\+ - * / %

除%外，都可用于整数和浮点数，%只能用于整数，返回余数

对于/, 若两个操作数都为整数，则执行整数除法，但只要有浮点数就执行浮点数除法

#### 移位操作符

不能移动负数位

<<	左移操作符（左移一位相当于*2）		移的是补码

\>>	右移操作符（右移一位相当于/2）：1.算术右移，右边丢弃，左边补原符号位，如正数补0，负数补1

​								2.逻辑右移，右边丢弃，左边补零

```c
int main() {
    // 10000000 00000000 00000000 00000001	源码
    // 11111111 11111111 11111111 11111110	反码
    // 11111111 11111111 11111111 11111111	补码
	int a = -1;
    // 11111111 11111111 11111111 11111111	移位后的补码
	int b = a >> 1;
	printf("%d\n", b);
	return 0;
}
```

#### 位操作符

&	按位与

|	按位或

^	按位异或

**不用临时变量交换两个变量**

```c
// 加减法 - 缺陷：整型的空间有限，相加不能超过整型所能表示的最大值，不能用于超级大的值，可能会溢出
int main() {
	int a = 3;		// 011
	int b = 5;		// 101
	// 加减法
	a = a + b;
	b = a - b;
	a = a - b;
	printf("%d %d", a, b);
	return 0;
}
// 异或操作
a = a ^ b;		// 110
b = a ^ b;		// 011	5
a = a ^ b;		// 101	3
```

#### 求整数存储在内存中的二进制中1的个数

```c
int main() {
	int num = 0;
	scanf("%d", &num);
	int count = 0;
	// 只能用于正数
	//// 统计num补码中1的个数
	//while (num) {
	//	if (num % 2 == 1) {
	//		count++;
	//	}
	//	num = num / 2;
	//}
	//printf("%d\n", count);

	// 与 1 按位与，如果等于1，则说明最后一位是 1，然后右移位
	int i = 0;
	for (i = 0; i < 32; i++) {
		if (1 == ((num >> i) & 1)) {
			count++;
		}
	}
	printf("%d\n", count);
	return 0;
}
```



```c
int main() {
	int a = 0;
	char b = 'q';
	int arr[10] = { 0 };

	printf("%d\n", sizeof(a));		// 4
	printf("%d\n", sizeof a);		// 4
	printf("%d\n", sizeof(int));	// 4

	printf("%d\n", sizeof(b));		// 1
	printf("%d\n", sizeof(char));	// 1

	printf("%d\n", sizeof(arr));	// 40
	printf("%d\n", sizeof(int [10]));	//40
	printf("%d\n", sizeof(int [5]));	// 20

	short s = 0;
	int a = 10;
	// 看被赋值的类型，里面的表达式不参与运算的，相当于sizeof(s)
	printf("%d\n", sizeof(s = s + a));		// 2
	printf("%d\n", s);		// 0

	return 0;
}
```



```c
前置++，先++， 后使用
后置++， 先使用， 后++
int main() {
	int a = 10;
	printf("%d\n", ++a);		// 11
	a = 10;
	printf("%d\n", a++);		// 10

	return 0;
}
```

#### 逻辑操作符

```c
int main() {
	int i = 0, a = 0, b = 2, c = 3, d = 4;
	// 因为前面为0，所以结果为0，不计算后面
	i = a++ && ++b && d++;		// 逻辑与，只要左边为0，就不计算右边	
	printf("a = %d\n b = %d\n c = %d d = %d", a, b, c, d);		// 1 2 3 4

	int i = 0, a = 0, b = 2, c = 3, d = 4;
	i = a++ || ++b || d++;		// 逻辑或， 只要左边为1，就不计算右边
	printf("a = %d\n b = %d\n c = %d d = %d", a, b, c, d);		// 1 3 3 4
	return 0;
}
```

#### 逗号表达式

从左到右依次执行，整个表达式的结果为最后一个表达式的结果

```c
int main() {
	int a = 1;
	int b = 2;
    // 0  12  12  13
	int c = (a > b, a = b + 10, a, b = a + 1);		// 13
	printf("%d\n", c);
	return 0;
}
```

#### 访问结构体成员

.	结构体.成员

->	结构体指针->成员

#### 隐式类型转换

整数运算总是至少以缺省整型类型的精度来进行的

为了获得这个精度，表达式中的**字符**和**短整型操作数**在使用前被转换为普通整形，这种转换

称为**整型提升**。

**整型提升**(参与运算的类型没有达到整型大小时，表达式中的char和short)：有符号数整型提升补的是符号位的数，如正数补0，负数补1

```c
int main() {
	// 00000000 00000000 00000000 00000011
	// 00000011
	char a = 3;
	// 00000000 00000000 00000000 01111111
	// 01111111
	char b = 127;
	// 相加要进行整型提升
	// a -- 00000000 00000000 00000000 00000011
	// b -- 00000000 00000000 00000000 01111111
	// 所以c -- 00000000 00000000 00000000 10000010
	// 然后再变回char，截断
	// c -- 10000010
	char c = a + b;
	// 打印再进行整型提升
	// c -- 11111111 11111111 11111111 10000010	--	补码
	// 11111111 11111111 11111111 10000001	-- 反码
	// 10000000 00000000 00000000 01111110	-- 源码
	printf("%d\n", c);		// -126
	return 0;
}
```

```c
int main() {
	char a = 0xb6;
	short b = 0xb600;
	int c = 0xb6000000;
	if (a == 0xb6)		// 进行整型提升，变了
		printf("a");
	if (b == 0xb600)	// 进行整型提升，变了
		printf("b");
	if (c == 0xb6000000)
		printf("c");		// 输出c
	return 0;
}
```

```c
// 参与表达式运算，进行整型提升
int main() {
	char c = 1;
	printf("%u\n", sizeof(c));		// 1
	printf("%u\n", sizeof(+c));		// 4 -- 整型提升
	printf("%u\n", sizeof(!c));		// 1
	return 0;
}
```

#### 算术转换





**指针类型决定了指针解引用操作时能访问的空间大小**



##### 野指针：

指针指向的位置不可知

1.指针未初始化

```c
int main() {
	// int a;		// 局部变量未初始化，默认为随机值
	int* p;		// 局部的指针变量，被初始化为随机值
	return 0;
}
```



2.指针越界访问

```c
int main() {
	int arr[10] = { 0 };
	int* p = arr;
	int i = 0;
	for (i = 0; i < 12; i++) {
		p++;		// 超过数组范围
	}
	return 0;
}
```



3.指针指向的空间释放

```c
int* test() {
	int a = 10;
	return &a;
}
int main() {
	int* p = test();	// 函数结束，a的空间被释放
	*p = 20;
	printf("%d\n", *p);
	return 0;
}
```



#### 指针运算

指针+-整数

指针-指针	--	得到中间元素的个数（指针要指向同一块空间）

指针的关系运算



##### 用指针算字符串长度

```c
my_strlen(char* str) {
	char* start = str;
	char* end = str;
	while (*end != '\0') {
		end++;
	}
	return end - start;
}
int main() {
	char arr[] = "bit";
	int len = my_strlen(arr);
	printf("%d\n", len);
	return 0;
}
```

