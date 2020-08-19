# 1、判断字符串是不是空白字符
# 空格、\t、\r、\n，则返回True，这些都是空白字符。
space_str = "  \t\n\r"
print(space_str.isspace())
# 2、判断是不是只包含字母
alpha_str = "abs"
print(alpha_str.isalpha())
# 3、判断是不是只包含字母或数字
alnum_str = "1"
print(alnum_str.isalnum())
# 4、判断是不是只包含数字
num_str = "123"
print(num_str.isdecimal())
# 5、判断是不是只包含数字、\u00b2、(1)
num_str = "\u00b2"
print(num_str.isdigit())
# 6、判断是不是只包含数字、汉字数字(二者可以同时包含)
num_str = "一123"
print(num_str.isnumeric())
# 7、判断是不是每个字母都首大写
words_str = "Hello Hello"
print(words_str.istitle())
# 8、判断是不是都是小写字母
words_str = "hello hello"
print(words_str.islower())
# 9、判断是不是都是大写字母
words_str = "HELLO HELLO"
print(words_str.isupper())

hello_str =  "hello world"
# 10、检查字符串是否以指定字符串开始
print(hello_str.startswith("hello"))
# 11、检查字符串以指定字符串结束
print(hello_str.endswith("world"))
# 12、查找指定的字符串，返回字符串开始的索引
# 使用index也可以找到开始的索引，如果指定字符串不存在会报错
# 而find会返回-1
print(hello_str.find("h"))
# 13、替换字符串
print(hello_str.replace("world","python"))
# replace返回一个新的字符串，不会修改原来的字符串
print(hello_str)

# 对齐
poem = ["登楼",
        "王之涣",
        "白日依山尽",
        "黄河入海流",
        "欲穷千里目",
        "更上一层楼"
]
for poem_str in poem:
    print("|%s|" % poem_str.rjust(10, " "))

# 去除空白字符
poem = ["\t\n登楼",
        "王之涣",
        "白日依山尽",
        "黄河入海流",
        "欲穷千里目",
        "更上一层楼"
]
for poem_str in poem:
    # 使用strip去除空白字符
    print("|%s|" % poem_str.strip().center(10, " "))
# print(poem)
# 拆分和连接字符串
str = "asbvfjk"
print(str.partition("bv"))
new_str = str.partition("bv")
print(" ".join(new_str))