class Cat:

    def drink(self):
        # 哪一个对象调用的方法，self就是哪一个对象的引用
        print("%s爱喝水" % self.name)

    def eat(self):
        print("小猫爱吃鱼")


# 创建对象

tom = Cat()
# 在类的外部增加对象属性
tom.name = "Tom"

tom.eat()
tom.drink()

# print(tom)
# addr = id(tom)
# print("%d" % addr)

# 再创建一个对象
lazy_cat = Cat()

lazy_cat.name = "大懒猫"
lazy_cat.eat()
lazy_cat.drink()
# print(lazy_cat)


