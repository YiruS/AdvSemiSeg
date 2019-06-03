from __future__ import absolute_import, division, print_function, unicode_literals

import optparse
import os
import sys
import pickle

import numpy as np

import utils.photometric_transforms as ph_transforms
from utils.generic_utils import parse_list, id_generator, get_free_gpu
from data_provider.eyeDataset import dataloader_dual
from helper_functions.trainer_gan import run_training_gan, run_testing_gan
from helper_functions.analysis import generate_result_images
from helper_functions.config import ReadConfig_GAN, augment_args_GAN
from models.model_utils import load_models

import torch
import torch.optim as optim

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
    parser.add_option("--lambda-ce", type="float",
                      dest="lambda_ce", default=1.0,
                      help="coefficient for CE loss")
    parser.add_option("--lambda-adv", type="float",
                      dest="lambda_adv", default=1.0,
                      help="coefficient for adversarial loss")
    parser.add_option("--checkpoint-SS",type="str",
                      dest="checkpoint_SS",default=None,
                      help="Path to pretrained model SS net")
    parser.add_option("--checkpoint-DNet", type="str",
                      dest="checkpoint_DNet", default=None,
                      help="Path to pretrained model Discriminator")
    parser.add_option("--train",action="store_true",
                      dest="train",default=False,
                      help="run training")
    parser.add_option("--val",action="store_true",
                      dest="val",default=False,
                      help="run validation",)
    parser.add_option("--test",action="store_true",
                      dest="test",default=False,
                      help="run testing (generate result images)",)

#### Read commandline arguments and network configuration fron input .ini file
    (args, opts) = parser.parse_args()

    if args.ini_file is None:
        print ('Model config file required, quitting...')
        sys.exit(0)

    GP, TP = ReadConfig_GAN(args.ini_file)
    args = augment_args_GAN(GP, TP, args)

#### Define CUDA boolean and transforms for input 2D images
    args.gpu = int(get_free_gpu())
    args.CUDA = args.gpu is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.transforms = [
        ph_transforms.ChangeBrightness(args.brightness_scale),
        ph_transforms.ToTensor(),
    ]

#### Define and load model
    model_SS = load_models(
        mode = "SS",
        device = device,
        args = args,
    )
    model_D = load_models(
        mode = "Discriminator",
        device = device,
        args = args,
    )
    optimizer_SS = optim.Adam(
        model_SS.parameters(),
        lr = args.lr_SS,
        betas=(0.9, 0.999),
        weight_decay=args.l2,
    )
    optimizer_SS.zero_grad()

    optimizer_D = optim.Adam(
        model_D.parameters(),
        lr = args.lr_G,
        betas=(0.9, 0.999),
        weight_decay=args.l2,
    )
    optimizer_D.zero_grad()

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr = args.lr,
    #     weight_decay = args.l2,
    # )

    print("Input Arguments: {}".format(args))

##### Load datasets
    if args.train:
        trainset_openeds, trainset_calipso, trainloader_openeds, trainloader_calipso = dataloader_dual(
            args,
            type="train",
        )
        class_weight_openeds = 1.0 / trainset_openeds.get_class_probability().to(device)
        criterion_CE_openeds = torch.nn.CrossEntropyLoss(
            weight = class_weight_openeds,
        ).to(device)

        class_weight_calipso = 1.0 / trainset_calipso.get_class_probability().to(device)
        criterion_CE_calipso = torch.nn.CrossEntropyLoss(
            weight = class_weight_calipso,
        ).to(device)

        criterion_ADV = torch.nn.BCEWithLogitsLoss().to(device)

        criterion_D = torch.nn.BCEWithLogitsLoss().to(device)


    ## load validation data set
    if args.train or args.val or args.finetune:
        valset_openeds, valset_calips, valloader_openeds, valloader_calipso = dataloader_dual(
            args,
            type = "val",
        )

##### main training loop
    if args.train:
        train_loss_ss_all, train_loss_d_all, val_loss_all = run_training_gan(
            trainloader_D1 = trainloader_openeds,
            trainloader_D2 = trainloader_calipso,
            valloader_D2 = valloader_calipso,
            model_SS = model_SS,
            model_D = model_D,
            criterion_CE_D1 = criterion_CE_openeds,
            criterion_CE_D2 = criterion_CE_calipso,
            criterion_ADV = criterion_ADV,
            criterion_D = criterion_D,
            optimizer_SS = optimizer_SS,
            optimizer_D = optimizer_D,
            device = device,
            args = args,
        )

        train_loss_SS = [t for t in train_loss_ss_all]
        train_loss_D = [t for t in train_loss_d_all]
        val_loss = [t for t in val_loss_all]

        try:
            o = open("%s/loss.pkl" % args.output_dir, "wb")
            pickle.dump([train_loss_SS, train_loss_D, val_loss], o, protocol=2)
            o.close()
        except FileNotFoundError as e:
            print (e)

    ## main validation loop to compute metrics on validation set if turned on
    if args.val:
        pm = run_testing_gan(
            valloader_D2 = valloader_calipso,
            model_SS = model_SS,
            device = device,
            args = args,
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
        testset_openeds, testset_calipso, testloader_openeds, testloader_calipso = dataloader_dual(
            args,
            type = "test",
        )

        pm_test, data_test, preds_test, acts_test = run_testing_gan(
            valloader_D2 = testloader_calipso,
            model_SS = model_SS,
            device = device,
            args = args,
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

