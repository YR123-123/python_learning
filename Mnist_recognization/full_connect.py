import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def full_connection():
    """
    用全连接来对手写数字进行识别
    """
    # 1、准备数据
    mnist = read_data_sets("./Mnist_data",one_hot=True)
    x = tf.placeholder(dtype=tf.float32,shape=(None,784))
    y_true = tf.placeholder(dtype=tf.float32,shape=(None,10))
    # 2、构建模型
    Weights = tf.Variable(initial_value=tf.random_normal(shape=(784,10)))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x,Weights) + bias

    # 3、构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # 4、优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    # 5、准确率计算
    equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
    accuracy = tf.reduce_mean(tf.cast(equal_list),tf.float32)
    # tf.cast用于将布尔型数据转换为数据类型，如int8、float32等
    # 初始化变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        image, label = mnist.train.next_batch(100)

        print("训练之前损失为%f" % (sess.run(error, feed_dict={x: image, y_true: label})))

        for i in range(500):
            _, loss, acc = sess.run([optimizer,error,accuracy], feed_dict={x: image, y_true: label})
            print("第%d轮训练后的损失为%f,准确度为%f" % (i+1, loss, acc))

    return None

if __name__ == "__main__":
    full_connection()





