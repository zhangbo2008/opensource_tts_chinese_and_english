from transformers import AutoTokenizer, DetrForObjectDetection

tokenizer = AutoTokenizer.from_pretrained("facebook/detr-resnet-50")

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

from transformers import AutoTokenizer, DetrForSegmentation

tokenizer = AutoTokenizer.from_pretrained("facebook/detr-resnet-50-panoptic")

model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")






from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes