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
        # step = 0
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
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        # steps=tf.Variable(0,name='steps',trainable=False)
        sess.run(tf.global_variables_initializer())
        sum_loss = 0.0
        saver = tf.train.Saver(max_to_keep=20)
        # ckpt = tf.train.get_checkpoint_state('/opt/model2/')
        # print(ckpt.model_checkpoint_path)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     print('load success')
        # else:
        #     print('load fail')
        #     exit()
        for iter in range(200):
            for index in range(1, 7):
                with open('minitrain1.pkl', 'rb') as f:
                    dataset = pickle.load(f)
                for line in dataset:
                    img=line[0]
                    lb=line[1]
                    feed_dict={hed.image:img,hed.label:lb}
                    #_,loss1,loss2,loss3,loss4,loss5,step,summary=sess.run([train_op,rcf.loss1,rcf.loss2,rcf.loss3,rcf.loss4,rcf.loss5,global_step,merged],feed_dict=feed_dict)
                    _,loss,step,summary=sess.run([train_op,hed.loss,global_step,merged],feed_dict=feed_dict)
                    
                    if step%50==0:
                        print('step = ',step,' loss1 = ',loss)

                    if step%1000==0:
                        with open('train.txt','r') as f:
                            data_dir=f.readlines()
                        for dir in data_dir:
                            img = dir.strip().split()[0]
                            name=dir.strip().split()[0]
                            gt=dir.strip().split()[1]
                            img=join(root,img)
                            gt=join(root,gt)
                            if isfile(img)==False or isfile(gt)==False:
                                print(img,gt)
                            img=cv2.imread(img).astype(np.float32)
                            gt=cv2.imread(gt,0).astype(np.float32)
                            if img.ndim==2:
                                img=img[:,:,np.newaxis]
                                img=np.repeat(img,3,2)
                            mean=[104.00699, 116.66877, 122.67892]
                            img=img[np.newaxis,:,:,:]
                            if gt.ndim!=2:
                                raise Exception('lable shape error')
                            if gt.shape[0]!=img.shape[1] or gt.shape[1]!=img.shape[2]:
                                raise Exception('label and image shape error')

                            gt=gt[np.newaxis,:,:,np.newaxis]

                            out1=sess.run([hed.out1],feed_dict={hed.image:img,hed.label:gt})
                            out1=out1[0]
                            print(out1.shape)
                            out1=out1[0,:,:,0]
                            # out2 = out2[0, :, :, 0]
                            # out3 = out3[0, :, :, 0]
                            # out4 = out4[0, :, :, 0]
                            # out5 = out5[0, :, :, 0]
                            #out = out[0, :, :, 0]
                            out1=255*(1-out1)
                            # out2 = 255 * (1 - out2)
                            # out3 = 255 * (1 - out3)
                            # out4 = 255 * (1 - out4)
                            # out5 = 255 * (1 - out5)
                            #out = 255 * (1 - out)
                            # print(out1.shape,out2.shape,out3.shape,out4.shape,out5.shape,out.shape)
                            cv2.imwrite('test/'+name[-11:-5]+'_out1_.png',out1)
                            # cv2.imwrite('test/'+name[-11:-5] + '_out2_.png', out2)
                            # cv2.imwrite('test/'+name[-11:-5] + '_out3_.png', out3)
                            # cv2.imwrite('test/'+name[-11:-5] + '_out4_.png', out4)
                            # cv2.imwrite('test/'+name[-11:-5] + '_out5_.png', out5)
                            # #cv2.imwrite('test/'+name[-11:-5] + '_out_.png', out)

                    if step%10000==0:
                        saver.save(sess, 'model/model.ckpt', global_step=global_step)
        saver.save(sess, 'model/model.ckpt', global_step=global_step)
        sess.close()


if __name__ == '__main__':
    main()
