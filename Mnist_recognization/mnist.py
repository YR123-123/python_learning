import tensorflow as tf
import  numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import  OneHotEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('X_train.shape:',X_train.shape)
print('X_test.shape:',X_test.shape)
print('y_train.shape:',Y_train.shape)
print('y_test.shape:',Y_test.shape)

# X_train.shape:(60000,28,28)
# X_test,shape:(10000,28,28)
# Y_train.shape:(60000,)
# Y_test.shape:(10000,)

#参数设置
learning_rate = 0.001
training_epochs = 50
batch_size = 100
display_step = 1

# 网络参数
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9，一共10类)

def onehot(y,star,end,categories='auto'):
    ohot = OneHotEncoder()
    a = np.linspace(star, end-1, end-star) # linspace用于产生（end-star）个数字，范围是star至end-1
    b = np.reshape(a,[-1,1]).astype(np.int32) # b=[0,1,2,3,4,5,6,7,8,9]
    ohot.fit(b)
    c = ohot.transform(y).toarray() # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
    # y原本的值label为0~9，经过transform操作变成onehot编码形式
    # 以y=array([[1],[3]])为例：array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])

    return  c


def MNISTLabel_TO_ONEHOT(X_train, Y_train, X_test, Y_test, shuff=True):
    Y_train = np.reshape(Y_train, [-1,1])  # reshape(-1,1)
    Y_test = np.reshape(Y_test, [-1,1])
    # Y进行onehot编码
    Y_train = onehot(Y_train.astype(np.int32), 0, n_classes)
    Y_test = onehot(Y_test.astype(np.int32), 0, n_classes)
    if shuff == True:
        X_train, Y_train = shuffle(X_train, Y_train)
        X_test, Y_test = shuffle(X_test, Y_test)
        return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = MNISTLabel_TO_ONEHOT(X_train, Y_train, X_test, Y_test)
# 以上完成了onhot和shuffle功能

# tf Graph input
# x = tf.placeholder(tf.float32, shape=(None, n_input)) # shape= [None, 784]
# y = tf.placeholder(tf.float32, shape=(None, n_classes))  # shape= [None, 10]
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases={
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),  # bias是一维的，也要用[]
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 构建模型
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

# 开启会话
with tf.Session() as sess:  # tf.Session()一定要有括号
    sess.run(init)

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        # 遍历整个数据集
        for i in range(total_batch):
            #    batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_x = X_train[i * batch_size:(i + 1) * batch_size,:]  # x初始维度为[60000,28,28],要修改成[60000,784]
            batch_x = np.reshape(batch_x,[-1,28*28])
            batch_y = Y_train[i * batch_size:(i + 1) * batch_size,:]
            correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
            Accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
            _, loss, acc = sess.run([optimizer, cost, Accuracy], feed_dict={x:batch_x, y:batch_y})

            # Compute average loss
            avg_cost += loss / total_batch

        # 显示每一轮训练的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=","{:.9f}".format(avg_cost), "Accuracy:", acc)
    print("Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    X_test = np.reshape(X_test, [-1, 28*28])
    print("Test_acc=",accuracy.eval({x:X_test, y:Y_test}))
    print(sess.run(tf.argmax(Y_test[:30],1)), "Real Number")
    print(sess.run(tf.argmax(pred[:30],1), feed_dict = {x:X_test,y:Y_test}), "Prediction_Number")




















