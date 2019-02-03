import tensorflow as tf

tf.enable_eager_execution()

a = tf.constant(1)
b = tf.constant(1)
c = tf.add(a, b)  # 也可以直接写 c = a + b，两者等价

print(c)

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)

x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))
# tf.constant_initializer 指定x的数据类型为 float 32
with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)  # 计算y关于x的导数
print([y.numpy(), y_grad.numpy()])

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.get_variable('w', shape=[2, 1], initializer=tf.constant_initializer([[1.], [2.]]))
b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer([1.]))
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])

# 烧包包的练习

# X = tf.constant([
#     [1., 2., 3.],
#     [4., 5., 6.],
#     [7., 8., 9.]
# ])
# w = tf.get_variable('w', shape=[3, 2], initializer=tf.constant_initializer([[3.] * 6]))
# b = tf.get_variable('b', shape=[2, 1], initializer=tf.constant_initializer([[3.] * 2]))
# with tf.GradientTape() as tape:
#     F = tf.matmul(tf.matmul(tf.matmul(X, X), w), b)
# w_g, b_g = tape.gradient(F, [w, b])
# print(F.numpy(), w_g.numpy(), b_g.numpy())

X = tf.constant(X)
y = tf.constant(y)

a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
variables = [a, b]

num_epoch = 1000
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
