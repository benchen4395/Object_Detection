###  YOLOv3_TensorFlow by BenChen:

### 必要的改进

>>>>>>> the modified of the weight files
1、将图像显示采用了plt,而不是cv2，保证图像在显示时也可以执行后面的修改；

2、对于数据的读取操作使用了tf.data管道，加速了模型的读入读出(我的硬件条件是2块GTX 1080)；

3、使用tf.image.non_max_suppression进行了非极大值抑制的GPU实现；

4、对核心代码进行了必要的注释。


windows10，单块1050Ti下，实时检测速度35fps

####有任何问题可以咨询：benchen4395@gmail.com

-------
Reference:

https://github.com/pjreddie/darknet

https://github.com/wizyoung/YOLOv3_TensorFlow

https://blog.csdn.net/leviopku/article/details/82660381

https://blog.csdn.net/leviopku/article/details/82660381

https://pjreddie.com/media/files/papers/YOLOv3.pdf




 
