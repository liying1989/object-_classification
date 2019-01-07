
import argparse

from train import train_net
import dataset
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#[ 避雷器，绝缘体，驱鸟器，防震鞭，防震锤，复合绝缘子，警示牌]
#classes = ['arrester', 'insulator','bird_protection','shockproof_whip','shockproof_hammer','composite_insulator','warning_signs']
#classes = ['insulator','composite_insulator']
classes = ['cats2','dogs2']
validation_size = 0.2

def parse_args():
    parser = argparse.ArgumentParser(description='image Classification')
    parser.add_argument('--data_dir', type=str, default='./data/train_data',
                        help='The path to the CIFAR-10 data directory.')
    parser.add_argument('--model_out_dir', type=str, default='./out/',
                        help='The directory where the model will be stored.')

    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard/test3/',
                        help='The directory where the tensorboard out will be stored.')

    parser.add_argument('--img_size', type=int, default=224,
                        help='The size of the model to use.')

    parser.add_argument('--num_channels', type=int, default=3,
                        help='The size of image channel.')


    parser.add_argument('--batch_size', type=int, default=32,
                        help='The number of images per batch.')
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default=['vgg16'], type=str)

    parser.add_argument('--max_iters',
                        default='8000', type=int)


    return parser.parse_args()



def main():
    args = parse_args()
    print("load data over")
    data = dataset.read_train_sets(args.data_dir, args.img_size, classes, validation_size = validation_size)

    train_net(args.net,
              data,
              args.img_size,
              args.num_channels,
              batch_size = args.batch_size,
              model_out_dir = args.model_out_dir,
              tensorboard_dir=args.tensorboard_dir,
              max_iters= args.max_iters)








if __name__ == '__main__':
    main()
