class Gun:

    def __init__(self, model):

        # 1.枪的型号
        self.model = model
        # 2.子弹的数量
        self.bullet_count = 0

    def add_bullet(self, count):

        self.bullet_count += count

    def shoot(self):

        # 1.判断子弹的数量
        if self.bullet_count <= 0:
            print("%s没有子弹了" % self.model)

            return

        # 2.发射子弹
        self.bullet_count -= 1

        # 3.提示发射信息

        print("[%s]突突突... [%d]" % (self.model, self.bullet_count))

class Soldier:

    def __init__(self, name):

        # 1.姓名
        self.name = name

        # 2.新兵没有枪
        self.gun = None

    def fire(self):

        # 1.判断士兵有没有枪

        # if self.gun == None:
        if self.gun is None:
            print("%s士兵还没有枪" % self.name)
            return

        # 2.高喊口号
        print("冲啊...[%s]" % self.name)

        # 3.装子弹
        self.gun.add_bullet(50)

        # 4.开火
        self.gun.shoot()

# 1.创建对象
ak47 = Gun("AK47")
# ak47.add_bullet(50)
# ak47.shoot()

# 2.创键许三多
xusanduo = Soldier("xusanduo")

xusanduo.gun = ak47

xusanduo.fire()

print(xusanduo.gun)