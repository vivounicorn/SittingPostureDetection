# coding:utf-8

import torch
from torch.nn import functional as F

from torchreid import metrics
import cv2
from torchvision import transforms
from torch.autograd import Variable
import torchreid
import os

'''
you should download osnet_ain_x1_0 model to '~/.cache/torch/checkpoint/'
'''


def reid_model():
    model = torchreid.models.build_model(
        name="osnet_ain_x1_0",
        num_classes=100,
        loss="softmax",
        pretrained=True
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    return model


class Compare(object):

    def __init__(self, model=None,
                 origin_img="./data/origin_image", normalize_feature=True):

        self.model = model
        self.origin_img_dir = origin_img
        self.is_normalize_f = normalize_feature

    def extract_feature(self, input_img):

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (128, 256))
        input_as_tensor = transforms.ToTensor()(input_img)

        input_as_tensor = input_as_tensor.unsqueeze(0)

        input_as_tensor = Variable(input_as_tensor)
        if torch.cuda.is_available():
            input_as_tensor = input_as_tensor.cuda()

        return self.model(input_as_tensor)

    def encode_image_dir(self):

        img_list = os.listdir(self.origin_img_dir)
        f_, name_ = [], []
        for img_name in img_list:
            img_path = os.path.join(self.origin_img_dir, img_name)
            img = cv2.imread(img_path)

            print(img_path)

            feature = self.extract_feature(self.model, img)
            feature = feature.data.cpu()

            f_.append(feature)
            name_.append(img_name.split(".")[0])

        f_ = torch.cat(f_, 0)

        if self.is_normalize_f:
            f_ = F.normalize(f_, p=2, dim=1)

        return f_, name_

    def run(self, compaer_img, origin_f, origin_name, dist_metric='cosine'):

        compare_f = self.extract_feature(compaer_img).data.cpu()

        if self.is_normalize_f:
            compare_f = F.normalize(compare_f, p=2, dim=1)

        distmat = metrics.compute_distance_matrix(compare_f, origin_f, metric=dist_metric)
        distmat = distmat.numpy()
        dist_list = distmat.tolist()[0]  # to list

        top_id = dist_list.index(min(dist_list))
        if min(dist_list) < 0.5:
            identify_name = origin_name[top_id]
        else:
            identify_name = "Unknow"

        return identify_name, min(dist_list)


def test():
    path1 = "/home/zhanglei/Gitlab/pytorch-reid/image/_origin"
    path2 = "/home/zhanglei/Gitlab/pytorch-reid/image/origin/Zhang HL4.jpg"

    compare = Compare(model=reid_model(), origin_img=path1)
    origin_f, origin_name = compare.encode_image_dir()

    compare_img = cv2.imread(path2)
    identify_name, score = compare.run(compare_img, origin_f, origin_name)
    print(identify_name, score)


if __name__ == "__main__":
    test()
