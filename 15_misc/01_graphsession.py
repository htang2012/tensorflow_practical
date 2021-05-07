import tensorflow.compat.v1 as tf
# 构建计算图g1
g1=tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量'v',并设置初始值为0。
    v=tf.get_variable('v',initializer=tf.zeros_initializer()(shape = [1]))
    
# 构建计算图g2
g2=tf.Graph()
with g2.as_default():
    # 在计算图g2中定义变量'v',并设置初始值微1。
    v=tf.get_variable('v',initializer=tf.ones_initializer()(shape = [1]))

# 在计算图g1中读取变量'v'的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))
        # 输出结果[0.]

# 在计算图g2中读取变量'v'的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('',reuse=True):
        print(sess.run(tf.get_variable('v')))
        # 输出结果[1.]。
