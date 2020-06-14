import tensorflow as tf
import os


def linear_regression():
    with tf.variable_scope("prepare_data"):
        # 1)准备数据
        x = tf.random_normal(shape=[100,1],name="features")
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    with tf.variable_scope("create_model"):
        # 2)构造模型
        weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]),name="weights")
        bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]),name="bias")
        y_predict = tf.matmul(x, weights) + bias

    with tf.variable_scope("loss_function"):
        # 3)构造损失函数
        error = tf.reduce_mean(tf.square(y_predict - y_true))

    with tf.variable_scope("optimizer"):
        # 4)优化损失
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 创建saver对象
    saver = tf.train.Saver()

    # 2、收集变量
    tf.summary.scalar("error",error)
    tf.summary.histogram("weights",weights)
    tf.summary.histogram("bias", bias)

    # 合并变量
    merged = tf.summary.merge_all()

    # 显示地初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)
        # 1、创建事件文件
        if not os.path.exists("./tmp/linear"):
            os.makedirs("./tmp/linear")
        file_writer = tf.summary.FileWriter("./tmp/linear", graph=sess.graph)

        # 查看初始化模型参数后的值
        print("训练前模型参数为：权重%f，偏置%f，损失%f" % (weights.eval(), bias.eval(), error.eval()))

        # 开始训练
        for i in range(100):
            sess.run(optimizer)
            print("第%d次训练后模型参数为：权重%f，偏置%f，损失%f" % (i,weights.eval(), bias.eval(), error.eval()))
            # 运行合并变量操作
            summary = sess.run(merged)
            file_writer.add_summary(summary,i)

            # 保存模型
            if i % 10 == 0:
                saver.save(sess, "./tmp/model/my_linear.ckpt")

        #加载模型
        # if os.path.exists("./tmp/model/checkpoint"):  # 这里检查的对象是checkpoint文件
        #     saver.restore(sess,"./tmp/model/my_linear.ckpt")


    return None





if __name__ == "__main__":
    # 实现线性回归
    linear_regression()
