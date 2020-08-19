class Cat:
    def __init__(self, new_name):
        print("这是一个初始化方法")

        # self.属性名 = 属性的初始值
        self.name = new_name

tom = Cat("Tom")
print(tom.name)

lazy_cat = Cat("大脸猫")
print(lazy_cat.name)