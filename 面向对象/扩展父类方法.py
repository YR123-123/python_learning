class Animal:

    def eat(self):
        print("吃")

    def drink(self):
        print("喝")

    def run(self):
        print("跑")

    def sleep(self):
        print("睡")

class Dog(Animal):

    def bark(self):
        print("汪汪叫")


class XiaoTtianQuan(Dog):

    def fly(self):
        print("我会飞")

    def bark(self):

        # 1.针对子类特有的需求，编写代码
        print("神一样的叫唤...")

        # 2.使用super(),调用原本在父类中封装的方法
        # super().bark()

        # 使用父类名.方法(self)
        Dog.bark(self)

        # 2.增加其他子类的代码
        print("#$%#^#^&*%&")

# 创键一个哮天犬对象
xtq = XiaoTtianQuan()
# xtq.fly()
xtq.bark()