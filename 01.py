import tensorflow as tf

# -----------------------------------------------------
# Define tensorflow model
# https://gist.github.com/eerwitt/518b0c9564e500b4b50f
# -----------------------------------------------------
# 定義 graph (tensor 和 flow)
str = "Hello Tensorflow"

final_image_array = []
# 執行 graph
with tf.Session() as sess:
    # 執行tensorflow時要先初始化(初學者照抄即可!)
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) # 建立多執行緒

    # exec tf flow
    print(str)

    # 停止多執行緒(初學者照抄前可!)
    coord.request_stop()
    coord.join(threads)