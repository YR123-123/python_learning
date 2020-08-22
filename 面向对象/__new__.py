class MusicPlayer:

    def __new__(cls, *args, **kwargs):

        # 1.创建对象时，new方法会自动调用
        print("创建对象，分配空间")

        # 2.为对象分配空间，调用父类方法
        instance = super().__new__(cls)

        # 3.返回对象的引用
        return instance

    def __init__(self):
        print("播放器初始化")

# 创建播放器对象
player = MusicPlayer()

print(player)