# TensorFlow-Java-Examples
Examples of Invoking TensorFlow from Java


## Java调用TensorFlow的两种方法

使用Java调用TensorFlow大致有两种方法：

+ 直接使用TensorFlow官方API调用训练好的pb模型: [https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)


+ (推荐) 使用KerasServer托管TensorFlow/Keras代码及模型:
[https://github.com/CrawlScript/KerasServer](https://github.com/CrawlScript/KerasServer)


虽然使用TensorFlow官方Java API可以直接对接训练好的pb模型，但在实际使用中，依然存在着与跨语种对接相关的繁琐代码。例如虽然已有使用Python编写好的基于TensorFlow的文本分类代码，但TensorFlow Java API的输入需要是量化的文本，这样我们又需要用Java重新实现在Python代码中已经实现的分词、从字符串到索引的转换等预处理操作（这些操作同时依赖于Python代码依赖的单词表等数据）。另外，由于Java没有numpy支持，在构建多维数组作为输入时，使用的依然是类似循环的操作，非常繁琐。


[KerasServer](https://github.com/CrawlScript/KerasServer)支持restful交互，因此可以支持用任何程序语言调用TensorFlow/Keras。由于KerasServer的服务端提供Python API, 因此可以直接将已有的TensorFlow/Keras Python代码和模型转换为KerasServer API，供Java/c/c++/C#/Python/NodeJS/Browser Javascript等调用，而不需要再其他语种中进行繁琐的数据预处理操作。例如，Java可直接将需要分类的文本数据提交给KerasServer，KerasServer可利用已有的Python代码对字符串进行分词、预处理等操作。



本教程介绍如何用TensorFlow官方Java API调用TensorFlow(Python)训练好的模型。教程的代码可在专知的Github项目中找到：[https://github.com/ZhuanZhiCode/TensorFlow-Java-Examples](https://github.com/ZhuanZhiCode/TensorFlow-Java-Examples)


## 依赖库

### Python依赖
TensorFlow

```bash
pip install tf-nightly
```


### Java依赖
本教程使用的是TensorFlow官方提供了Java接口，因此我们需要导入下面的Maven依赖：

```xml
<dependency>
   <groupId>org.tensorflow</groupId>
   <artifactId>tensorflow</artifactId>
   <version>1.5.0</version>
</dependency>
```

此外，还有一些工具类依赖：
```xml
<dependency>
   <groupId>commons-io</groupId>
   <artifactId>commons-io</artifactId>
   <version>2.6</version>
</dependency>
```

## 保存pb模型

下面的代码中，x是图的输入，z是图的输出。在代码的最后，调用`tf.graph_util.convert_variables_to_constants`将图进行转换，最后将图保存为模型文件(pb)。

```python
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
```

## 在Java中调用TensorFlow的图(pb模型)

模型的执行与Python类似，依然是导入图，建立Session，指定输入(feed)和输出(fetch)。

```java
import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.FileInputStream;
import java.io.IOException;

public class DemoImportGraph {

    public static void main(String[] args) throws IOException {
        try (Graph graph = new Graph()) {
            //导入图
            byte[] graphBytes = IOUtils.toByteArray(new FileInputStream("model.pb"));
            graph.importGraphDef(graphBytes);

            //根据图建立Session
            try(Session session = new Session(graph)){
                //相当于TensorFlow Python中的sess.run(z, feed_dict = {'x': 10.0})
                float z = session.runner()
                        .feed("x", Tensor.create(10.0f))
                        .fetch("z").run().get(0).floatValue();
                System.out.println("z = " + z);
            }
        }

    }
}
```

运行结果：
```
z = 2.9957323
```
