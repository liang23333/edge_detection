import pickle
import tensorflow as tf
from HED import HED
from os.path import join, isfile
import numpy as np
import cv2
import random


def main():
    tf.reset_default_graph()
    with tf.device('/gpu:0'):
        root = '/opt/HED-BSDS/'
        step = 0
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        hed=HED()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        print(type(global_step))
        learning_rate = tf.train.exponential_decay(1e-6, global_step, decay_steps=200000, decay_rate=0.1, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        print('start:')
        grads_and_vars = optimizer.compute_gradients(hed.loss_sum)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        sum_loss = 0.0
        saver = tf.train.Saver(max_to_keep=20)
        ckpt = tf.train.get_checkpoint_state('model/')
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load success')
        else:
            print('load fail')
            exit()
        with open('minitrain1.pkl','rb') as f:
            test=pickle.load(f)
        for one in test:
            feed_dict={hed.image:one[0],hed.label:one[1]}
            out=sess.run([hed.out],feed_dict=feed_dict)
            out=out[0][0]
            print(out.shape)
            out=255*(1-out)
            cv2.imwrite(str(step)+'.png',out)
        sess.close()


if __name__ == '__main__':
    main()
