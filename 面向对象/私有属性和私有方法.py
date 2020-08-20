class Woman:

    def __init__(self, name, age):

        self.name = name
        self.__age = age

    def __secret(self):
        # 在对象的内部是可以访问对象的私有属性的
        print("%s的年龄是%d" % (self.name, self.__age))


xiaofang = Woman("xiaofang", 18)
# print(xiaofang.__age)  # 私有属性在外界不能被直接访问
# xiaofang.__secret()  # 私有方法同样不能在外界被直接访问

# 伪私有属性和方法
print(xiaofang._Woman__age)
xiaofang._Woman__secret()