# Keras 学习

## GitHub
- [GitHub for Keras](https://github.com/keras-team/keras)

## 教程
- [Keras 中文文档](https://keras-cn.readthedocs.io/)
- [Deep Learning with Pyton](https://cnbeining.github.io/deep-learning-with-python-cn/3-multi-layer-perceptrons/ch10-project-multiclass-classification-of-flower-species.html) 有介绍更多关于Keras的东西及如何使用Keras 进行建模

## Blogs & 案例
- [AutoEncoders with Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Keras 案例 in kesci(K-Lab)](https://www.kesci.com/apps/home/user/profile/599b9afac8d2787da4d2914c)
- [K-Lab projects](https://www.kesci.com/apps/home/project)


## More Deep Learning Info
- [deeplearninggallery](http://deeplearninggallery.com/)
- [Open-CV](https://www.learnopencv.com/) 图像处理


## 深度学习中的 激活函数 与 损失函数
1. 用sigmoid作为激活函数，为什么往往损失函数选用binary_crossentropy 
	参考地址:https://blog.csdn.net/wtq1993/article/details/51741471
2. softmax与categorical_crossentropy的关系，以及sigmoid与bianry_crossentropy的关系。 
	参考地址:https://www.zhihu.com/question/36307214

3. 各大损失函数的定义:MSE,MAE,MAPE,hinge,squad_hinge,binary_crossentropy等 
	参考地址:https://www.cnblogs.com/laurdawn/p/5841192.html

> **binary_crossentropy** vs **categorical_crossentropy**
> **binary_crossentropy** 一般用作 **二分类**任务
> **categorical_crossentropy** 一般用作  **多分类**任务, 并伴随 **softmax** 为激活函数使用!