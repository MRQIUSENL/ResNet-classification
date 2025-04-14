import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 创建中英文对照字典
    flower_names = {
        "daisy": "雏菊",
        "dandelion": "蒲公英",
        "roses": "玫瑰",
        "sunflowers": "向日葵",
        "tulips": "郁金香"
    }

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
        # 将类别转换为中文
        class_indict = {k: flower_names[v] for k, v in class_indict.items()}

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 随机选择验证集中的三张图片
    val_root = r"F:\deep-learning-for-image-processing-master-master\pytorch_classification\Test5_resnet\flower_data\val"
    flower_classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    import random
    
    # 从每个类别文件夹中获取所有图片
    all_images = []
    for flower in flower_classes:
        flower_path = os.path.join(val_root, flower)
        if os.path.exists(flower_path):
            images = [os.path.join(flower_path, img) for img in os.listdir(flower_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(images)
    
    # 随机选择3张图片
    selected_images = random.sample(all_images, 3)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建一个1x3的子图布局
    plt.figure(figsize=(15, 5))
    
    # 对每张图片进行预测
    for idx, img_path in enumerate(selected_images, 1):
        print(f"\n预测图片: {os.path.basename(img_path)}")
        img = Image.open(img_path)
        
        plt.subplot(1, 3, idx)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        
        # prediction
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            
        # 修改标题显示格式
        print_res = f"{class_indict[str(predict_cla)]}\n概率: {predict[predict_cla].numpy():.3f}"
        plt.title(print_res, fontsize=12, pad=10)
        for i in range(len(predict)):
            print("类别: {:10}   概率: {:.3}".format(class_indict[str(i)],
                                                predict[i].numpy()))
    
    plt.tight_layout()  # 自动调整子图布局
    plt.show()


if __name__ == '__main__':
    main()
