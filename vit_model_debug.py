
'''
python模块中requests参数stream
PS:这个参数真没用过

当下载大的文件的时候，建议使用strea模式．

默认情况下是false，他会立即开始下载文件并存放到内存当中，倘若文件过大就会导致内存不足的情况．

当把get函数的stream参数设置成True时，它不会立即开始下载，当你使用iter_content或iter_lines遍历内容或访问内容属性时才开始下载。需要注意一点：文件没有下载之前，它也需要保持连接。

iter_content：一块一块的遍历要下载的内容
iter_lines：一行一行的遍历要下载的内容
使用上面两个函数下载大文件可以防止占用过多的内存，因为每次只下载小部分数据。
'''


from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')# 先进入特征提取器进行玩. 就是对于图片进行预处理!!!!!!!!!!!
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


# 2021-06-03,16点59   over debug!