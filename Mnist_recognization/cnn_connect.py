import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import  os

def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape))

def create_model(x):
    # 构建卷积神经网络
    y_predict = 0
    # 1) 第一个卷积大层
    with tf.variable_scope("conv1"):
        # 卷积层
        # 将x(None, 784)进行修改
        input_x = tf.reshape(x, shape = [-1, 28, 28 , 1])
        # 定义filter和偏置
        conv1_weights = create_weights(shape=[5,5,1,32])
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=input_x, filter=conv1_weights,strides=[1,1,1,1],padding="SAME") + conv1_bias
        # 激活层
        relu1_x = tf.nn.relu(conv1_x)
        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1,2,2,1], strides=[1,2,2,3],padding="SAME")
    # 2) 第二个卷积大层
    with tf.variable_scope("conv2"):
        # 卷积层
        # 定义filter和偏置
        conv2_weights = create_weights(shape=[5,5,32,64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights,strides=[1,1,1,1],padding="SAME") + conv2_bias
        # 激活层
        relu2_x = tf.nn.relu(conv2_x)
        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1,2,2,1], strides=[1,2,2,3],padding="SAME")
    # 3） 全连接层
    with tf.variable_scope("full_connection"):
        # [None, 7, 7, 64]变为[None, 7*7*64]
        # [None, 7*7*64] * [7*7*64,10] = [None, 10]
        x_fc = tf.reshape(pool2_x, shape=[None, 7*7*64])
        weights_fc = create_weights(shape=[7*7*64,10])
        bias_fc = create_weights(shape=[10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc


    return y_predict

def cnn_connection():
    """
    用全连接来对手写数字进行识别
    特征值：[None, 784]
    目标值：one_hot编码[None, 10]
    """
    # 1、准备数据
    with tf.variable_scope("mnist_data"):
        mnist = read_data_sets("./Mnist_data",one_hot=True)
        x = tf.placeholder(dtype=tf.float32,shape=(None,784))
        y_true = tf.placeholder(dtype=tf.float32,shape=(None,10))
    # # 全连接层神经网络计算
    # # 类别：10个类别  全连接层：10个神经元
    # # 参数：w[789,10] b[10]
    # # 随机初始化权重偏置参数，这些是优化的参数，必须用变量op定义
    # # 2、构建模型
    # with tf.variable_scope("model"):
    #     Weights = tf.Variable(initial_value=tf.random_normal(shape=(784,10)))
    #     bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    #     y_predict = tf.matmul(x,Weights) + bias

    y_predict = create_model(x)

    # 3、构造损失函数
    with tf.variable_scope("softmax_crossentropy"):
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # 4、优化损失
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    # 5、准确率计算
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list),tf.float32)
        # tf.cast用于将布尔型数据转换为数据类型，如int8、float32等

    tf.summary.scalar("loss", error)
    tf.summary.scalar("acc", accuracy)
    # tf.summary.histogram("weights", Weights)
    # tf.summary.histogram("biases", bias)
    merged = tf.summary.merge_all()

    # 初始化变量
    init = tf.global_variables_initializer()

    # 保存模型
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        if not os.path.exists("./tmp/cnn_mnist"):
            os.makedirs("./tmp/cnn_mnist")
        if not os.path.exists("./tmp/model_mnist"):
            os.makedirs("./tmp/model_mnist")
        file_writer = tf.summary.FileWriter("./tmp/cnn_mnist", graph=sess.graph)
        image, label = mnist.train.next_batch(100)

        print("训练之前损失为%f" % (sess.run(error, feed_dict={x: image, y_true: label})))

        for i in range(500):
            _, loss, acc = sess.run([optimizer,error,accuracy], feed_dict={x: image, y_true: label})
            summary = sess.run(merged, feed_dict={x: image, y_true: label})
            file_writer.add_summary(summary, i)
            print("第%d轮训练后的损失为%f,准确度为%f" % (i+1, loss, acc))
            if i % 10 == 0:
                saver.save(sess,"./tmp/model_mnist")

    return None

if __name__ == "__main__":
    cnn_connection()





