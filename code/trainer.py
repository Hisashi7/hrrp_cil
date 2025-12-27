import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import datetime
import time


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    all_seed = 1
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    dirName = r"logs/{}/{}/{}/{}/{}".format(
        args["prefix"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["model_name"]
    )
    baseDir = os.path.dirname(__file__)
    dirName = os.path.join(baseDir, dirName)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    logfilename = os.path.join(dirName, "Seed={}_Time={}.log".format(
        args["seed"],
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    ))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(all_seed)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args,
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)
    # if args["pretrain"]:
    #     begin_task = 1
    #     model._cur_task = 0
    #     model._known_classes = args["init_cls"]
    # else:
    begin_task = 0

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    # 计算整个过程训练时间——起始时间
    all_process_time_start = time.time()
    Average_task_training_time = 0

    for task in range(begin_task, data_manager.nb_tasks, 1):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        single_task_time_start = time.time()
        model.incremental_train(data_manager)
        sigle_task_time_end = time.time()
        single_task_training_time = sigle_task_time_end - single_task_time_start
        logging.info(
            "task:{} training time:{:.2f}s".format(task, single_task_training_time)
        )
        Average_task_training_time = (Average_task_training_time * task + single_task_training_time) / (task + 1)
        

        logging.info("All params: {}".format(count_parameters(model._network)))
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)


            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            logging.info('Average Accuracy (CNN): {}'.format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info('Average Accuracy (NME): {}'.format(sum(nme_curve["top1"])/len(nme_curve["top1"])))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            logging.info('Average Accuracy (CNN): {}'.format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    # 计算整个过程训练时间——结束时间
    all_process_time_end = time.time()
    logging.info("Time consumed in all training process:{:.2f}s".format(all_process_time_end - all_process_time_start))
    logging.info("Average Time consumed in single task:{:.2f}s".format(Average_task_training_time))

    #按照saved_pth/dataset/model_name/logname.pth保存model.static_state，并logging保存成功消息
    # save_dir = os.path.join("saved_pth", args["dataset"], args["model_name"])
    # os.makedirs(save_dir, exist_ok=True)

    # # Extract log name from logfilename
    # log_name = os.path.basename(logfilename)
    # save_path = os.path.join(save_dir, log_name + ".pth")

    # try:
    #     torch.save(model._network, save_path)
    #     logging.info(f"Model state dict saved successfully at: {save_path}")
    # except Exception as e:
    #     logging.error(f"Failed to save model state dict: {str(e)}")

    if len(cnn_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        logging.info('Accuracy Matrix (CNN):')
        logging.info("\n{}".format(np_acctable))
        logging.info('Forgetting (CNN): {}'.format(forgetting))
    if len(nme_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        logging.info('Accuracy Matrix (NME):')
        logging.info("\n{}".format(np_acctable))
        logging.info('Forgetting (NME): {}'.format(forgetting))
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
