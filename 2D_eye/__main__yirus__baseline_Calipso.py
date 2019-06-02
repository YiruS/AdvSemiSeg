from __future__ import absolute_import, division, print_function, unicode_literals

import optparse
import os
import sys
import pickle

import numpy as np
import torch.optim

import utils.photometric_transforms as ph_transforms
from utils.generic_utils import parse_list, id_generator, get_free_gpu
from data_provider.eyeDataset import dataloader
from helper_functions.torchsummary import summary
from helper_functions.trainer import run_training, run_testing, run_prediction, run_finetune
from helper_functions.trainer_utils import SoftDiceLoss, define_criterion
from helper_functions.analysis import generate_result_images
from helper_functions.config import ReadConfig,augment_args
from models.model_utils import load_display_model

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--data-root", type="str",
                      dest="data_root",
                      help="list of Path to data folder",)
    parser.add_option("--save-dir", type="str",
                      dest="save_dir",
                      help="Path to folder where model is saved")
    parser.add_option("--output-dir", type="str",
                      dest="output_dir", default=None,
                      help="Path to folder where output images are saved")
    parser.add_option("--ini-file",type="str",dest="ini_file",
                      default=None,
                      help="configuration initialization file; See for example ./Ini_Files")
    parser.add_option("--image-size", type="int",
                      dest="image_size", default=184,
                      help="image_size scalar (currently support square images)")
    parser.add_option("--checkpoint",type="str",
                      dest="checkpoint",default=None,
                      help="Path to pretrained model")
    parser.add_option("--train",action="store_true",
                      dest="train",default=False,
                      help="run training")
    parser.add_option("--val",action="store_true",
                      dest="val",default=False,
                      help="run validation",)
    parser.add_option("--test",action="store_true",
                      dest="test",default=False,
                      help="run testing (generate result images)",)
    parser.add_option("--finetune", action="store_true",
                      dest="finetune", default=False,
                      help="finetune model", )


#### Read commandline arguments and network configuration fron input .ini file
    (args, opts) = parser.parse_args()

    if args.ini_file is None:
        print ('Model config file required, quitting...')
        sys.exit(0)

    GP, TP = ReadConfig(args.ini_file)
    args = augment_args(GP, TP, args)

#### Define CUDA boolean and transforms for input 2D images
    args.gpu = int(get_free_gpu())
    args.CUDA = args.gpu is not None
    args.transforms = [
        ph_transforms.ChangeBrightness(args.brightness_scale),
        ph_transforms.ToTensor(),
    ]
#### Define and load model
    model = load_display_model(args)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = args.lr,
        weight_decay = args.l2,
    )

    print("Input Arguments: {}".format(args))

##### Load datasets
    if args.train or args.finetune:
        train_set, train_dataloader = dataloader(
            args,
            type="train",
        )
        class_weight, criterion = define_criterion(
            train_set,
            args,
        )

    ## load validation data set
    if args.train or args.val or args.finetune:
        val_set, val_dataloader = dataloader(
            args,
            type = "val",
        )

##### main training loop
    if args.train:
        train_loss_tensor, val_loss_tensor = run_training(
            train_dataloader,
            val_dataloader,
            model,
            criterion,
            optimizer,
            args,
        )
        train_loss = [t for t in train_loss_tensor]
        val_loss = [t for t in val_loss_tensor]
        # train_loss = [t.detach().cpu().numpy() for t in train_loss_tensor]
        # val_loss = [t.detach().cpu().numpy() for t in val_loss_tensor]

        try:
            o = open("%s/loss.pkl" % args.output_dir, "wb")
            pickle.dump([train_loss, val_loss], o, protocol=2)
            o.close()
        except FileNotFoundError as e:
            print (e)

    ## main validation loop to compute metrics on validation set if turned on
    if args.val:
        pm = run_testing(
            val_dataloader,
            model,
            args,
        )
        print('Global Mean Accuracy:', np.array(pm.GA).mean())
        print('Mean IOU:', np.array(pm.IOU).mean())
        try:
            o = open("%s/metrics.pkl" % args.output_dir, "wb")
            pickle.dump([pm.GA,pm.CA, pm.Precision, pm.Recall, pm.F1, pm.IOU],o,protocol=2)
            o.close()
        except FileNotFoundError as e:
            print (e)

#### main testing loop to compute metrics on test data as well as prediction segmented images
    if args.test:
        test_set, test_dataloader = dataloader(
            args,
            type = "test",
        )
        pm_test, data_test, preds_test, acts_test = run_testing(
            test_dataloader,
            model,
            args,
            get_images = True,
        )
        print('Global Mean Accuracy:', np.array(pm_test.GA).mean())
        print('Mean IOU:', np.array(pm_test.IOU).mean())
        print('Mean Recall:', np.array(pm_test.Recall).mean())
        print('Mean Precision:', np.array(pm_test.Precision).mean())
        print('Mean F1:', np.array(pm_test.F1).mean())

        try:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
        except:
            print('Output directory not specified')

        print("Generating predicted images ...")
        for i in range(len(data_test)):
            generate_result_images(
                input_image = data_test[i],
                target_image = acts_test[i],
                pred_image = preds_test[i],
                args=args,
                iou = pm_test.IOU[i],
                count = i,
            )

    if args.finetune:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))

        # # make optimizer only for certain layer
        # optimizer = torch.optim.Adam(
        #     model.layer.parameters(),
        #     lr=args.lr,
        #     weight_decay=args.l2,
        # )

        train_loss_tensor, val_loss_tensor = run_finetune(
            train_dataloader,
            val_dataloader,
            model,
            criterion,
            optimizer,
            args,
        )
        train_loss = [t for t in train_loss_tensor]
        val_loss = [t for t in val_loss_tensor]

        try:
            o = open("%s/loss.pkl" % args.output_dir, "wb")
            pickle.dump([train_loss, val_loss], o, protocol=2)
            o.close()
        except FileNotFoundError as e:
            print(e)
