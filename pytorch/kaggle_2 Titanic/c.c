#include <stdio.h>

int main()
{
    char arr[26];
    int i;

    for (i = 0; i < 26; i++)
    {
        scanf(" %c", &arr[i], 1);  // 加上 & 和 缓冲区大小，前面加空格跳过换行符
    }

    for (i = 0; i < 26; i++)
    {
        printf("%c", arr[i]);
    }

    return 0;
}
