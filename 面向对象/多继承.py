class A:

    def test(self):
        print("A --- test方法")

    def demo(self):
        print("A --- demo方法")

class B:

    def test(self):
        print("B --- test方法")

    def demo(self):
        print("B --- demo方法")

class C(A, B):
    # 多继承可以使得子类对象同时具有多个父类的属性和方法
    pass

# 创建子类对象
c = C()
c.test()
c.demo()
print(C.__mro__)
