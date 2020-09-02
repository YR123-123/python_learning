def demo1():
    return int(input("请输入一个整数："))


def demo2():
    return demo1()

# 利用异常的传递性，在主程序中捕获异常
# 在子函数中使用异常捕获，代码太繁琐
try:
    print(demo2())
except Exception as result:
    print("位置错误 %s" % result)