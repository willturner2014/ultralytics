

# 模型配置文件
model_yaml_path = r"D:\Codes\Python\0_Work\yolov11\ultralytics\cfg\models_my\yolo11s.yaml"
#数据集配置文件
data_yaml_path = r'D:\Codes\Python\0_Work\yolov11\ultralytics\cfg\datasets_my\rips_rs_hainan.yaml'
#预训练模型
pre_model_name = 'yolo11s.pt'
pre_model_path = r'D:\Codes\Python\0_Work\yolov11\ultralytics\pt\yolo11s.pt'


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    # 如何切换模型版本, 上面的ymal文件可以改为 yolo11s.yaml就是使用的11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    model.load(pre_model_path) # 是否加载预训练权重
    model.train(data=data_yaml_path,
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=True,  # 是否是单类别检测
                batch=16,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )
 
