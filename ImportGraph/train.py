#coding=utf-8
import tensorflow as tf


# 定义图
x = tf.placeholder(tf.float32, name="x")
y = tf.get_variable("y", initializer=10.0)
z = tf.log(x + y, name="z")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 进行一些训练代码，此处省略
    # xxxxxxxxxxxx

    # 显示图中的节点
    print([n.name for n in sess.graph.as_graph_def().node])
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names=["z"])

    # 保存图为pb文件
    with open('model.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())