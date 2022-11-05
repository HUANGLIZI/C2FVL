import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
#import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.C2FVL import C2FVL
from utils import *
import cv2
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output,py1,py2,py3,py4 = model(input_img.cuda(), text.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')
    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    if config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "QaTa_COVID":
        test_num = 2113
        model_type = config.model_name
        model_path = "./QaTa_COVID/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    elif config.task_name == "MosMedData+":
        test_num = 273
        model_type = config.model_name
        model_path = "./MosMedData+/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'C2FVL':
        config_vit = config.get_CTranS_config()
        model = C2FVL(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'C2FVL_pretrain':
        config_vit = config.get_CTranS_config()
        model = C2FVL(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text=[]
    test_bt,test_n,test_lu,test_lm,test_ll,test_ru,test_rm,test_rl = read_attr(config.test_dataset + 'Test_text.xlsx')
        #print('train_text:',train_bt,train_num,train_lu,train_lm,train_ll,train_ru,train_rm,train_rl )
    test_text.append(test_bt)
    test_text.append(test_n)
    test_text.append(test_lu)
    test_text.append(test_lm)
    test_text.append(test_ll)
    test_text.append(test_ru)
    test_text.append(test_rm)
    test_text.append(test_rl)
    #test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # print(names)
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, test_text, None, lab,
                                                           vis_path + str(names),
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
