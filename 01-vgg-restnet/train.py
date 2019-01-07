
try:
  import cPickle as pickle
except ImportError:
  import pickle
import tensorflow as tf
import os
import numpy as np
import glob
from utils.timer import Timer
from nets.vgg16 import vgg16
from nets.resnet import resnet


class SolverWrapper(object):

    def __init__(self,network, batch_size,data,model_output_dir,tensor_out_dir):
        self.net = network

        self.batch_size = batch_size

        self.pretrained_model = None
        self.last_snapshot_iter = 0
        self.saver = None
        self.train_data = data.train
        self.valid_data = data.valid
        self.output_dir = model_output_dir
        self.is_load = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.tensor_out_dir = tensor_out_dir
        if not os.path.exists(self.tensor_out_dir):
            os.makedirs(self.tensor_out_dir)

    def load_pretrained_model(self, data_path, session,  ignore_missing=False):
        if data_path.endswith('.ckpt'):
            self.saver = tf.train.import_meta_graph(data_path+'.meta')
            self.saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model " + subkey + " to " + key)
                        except ValueError:
                            print("ignore " + key)
                            if not ignore_missing:
                                raise
    #store snapshot
    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot

        filename = self.net.net_name + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))


    def load_from_snapshot(self):
        prefilename = self.net.net_name + '_iter_'

        pre_path = os.path.join(self.output_dir,prefilename)

        path = os.path.join(self.output_dir, '*.index')
        files = glob.glob(path)
        if len(files) == 0:
            return
        num_list = []
        for fl in files:
            num_list.append(int(fl[len(pre_path):].split('.')[0]))
        #print(num_list)
        self.last_snapshot_iter  = max(num_list)
        print('last_last_snapshot_iter is {}'.format(self.last_snapshot_iter))
        self.pretrained_model = pre_path +str(self.last_snapshot_iter)+ '.ckpt'





    def initialize(self, sess):

        self.train_data.reset_epoth()
        self.valid_data.reset_epoth()

        init = tf.global_variables_initializer()
        sess.run(init)

        self.load_from_snapshot()

        if self.pretrained_model is not None:
            print(('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model))
            self.load_pretrained_model(self.pretrained_model, sess, True)
            self.is_load = True



    def train_model(self,sess,learning_rate,max_iters):
        # input image tensorboard
        #tf.summary.image('input', self.net.x, 3)
        out_labls_num = self.train_data.num_lable
        y_true = tf.placeholder(tf.float32, shape=[None,out_labls_num], name='y_true')


        model_out = self.net.buildModel()
        logits = tf.nn.softmax(model_out, name="softmax")

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=y_true)
            cost = tf.reduce_mean(cross_entropy)

            loss_summary = tf.summary.scalar("loss",cost)
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        with tf.name_scope("accuracy"):
           # y_pred = tf.nn.softmax(logits, name='y_pred')
            y_pred_cls = tf.argmax(logits, dimension=1)
            y_true_cls = tf.argmax(y_true, dimension=1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            acc_summary = tf.summary.scalar("accuracy", accuracy)

        self.initialize(sess)

        #tensorboard
        writer = tf.summary.FileWriter(self.tensor_out_dir )
        writer.add_graph(sess.graph)
        #if not self.is_load:
        #merged = tf.summary.merge_all()
        merged = tf.summary.merge([loss_summary,acc_summary])

        timer = Timer()
        iter = self.last_snapshot_iter + 1

        while iter < max_iters+ 1:
            timer.tic()
            x_batch, y_true_batch = self.train_data.next_batch(self.batch_size)
            x_valid_batch, y_valid_batch = self.valid_data.next_batch(self.batch_size)
            feed_dict_tr = {self.net.x: x_batch, y_true: y_true_batch}


            if iter % int(25) == 0:

                epoch = int(iter / int(self.train_data.num_examples() / self.batch_size))
                feed_dict_val = {self.net.x: x_valid_batch, y_true: y_valid_batch}
                acc = sess.run(accuracy, feed_dict=feed_dict_tr)
                [val_loss,val_acc] = sess.run([cost,accuracy], feed_dict=feed_dict_val)
                msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
                print(msg.format(epoch + 1, iter, acc, val_acc, val_loss))
                print('speed: {:.3f}s / iter'.format(timer.average_time))
                if self.last_snapshot_iter != iter:
                    self.snapshot(sess, iter)

            summary, _ = sess.run([merged,optimizer], feed_dict=feed_dict_tr)
            writer.add_summary(summary, iter)
            timer.toc()
            iter += 1

def make_hparam_string(learning_rate):
    return "lr_%.0E"% ( learning_rate)



def train_net(nets,data,img_size,num_channels,batch_size,model_out_dir,tensorboard_dir = './tensorboard/',max_iters= 8000):


    tfconfig = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    tfconfig.gpu_options.allow_growth = True
    for learning_rate in [1e-3, 1e-4]:
        for i in range(len(nets)):

            tf.reset_default_graph()
            x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
            out_class_num = data.train.num_lable
            if nets[i] == 'vgg16':
                net = vgg16(x, out_class_num)
                output_dir = model_out_dir+'vgg/'
            elif nets[i] == 'res50':
                net = resnet(50,x, out_class_num)
                output_dir = model_out_dir+'res50/'
            elif nets[i] == 'res101':
                net = resnet(101,x, out_class_num)
                output_dir = model_out_dir+'res101/'
            elif nets[i] == 'res152':
                net = resnet(152,x,out_class_num)
                output_dir = model_out_dir+'res152/'
            else:
                raise NotImplementedError



            sess = tf.Session(config=tfconfig)
            print('Solving...')
            hparam = make_hparam_string(learning_rate)
            model_output_dir = output_dir + hparam
            print("Start run for %s" % hparam)
            print("model output path is %s" % model_output_dir)
            tensor_out_dir = tensorboard_dir + nets[i] + '_' + hparam
            sw = SolverWrapper(net, batch_size, data, model_output_dir, tensor_out_dir)
            sw.train_model(sess, learning_rate, max_iters)
            sess.close()
    print('done solving')
