from PIL import Image
from ultralytics import YOLO
from relsgg import RelateAnything

image = Image.open("assets/reel/images/horse.jpg").convert("RGB")

# 1. Find the objects. Any detector or segmenter works; this one is YOLO26.
detections = YOLO("yolo26m.pt")(image, conf=0.4, verbose=False)[0]
boxes = detections.boxes.xyxy.cpu().numpy()  # [N, 4]: x1, y1, x2, y2 in pixels
labels = [detections.names[int(c)] for c in detections.boxes.cls]  # for display only

# 2. Predict the relations between them.
model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", device="cuda")
for triplet in model.predict(image, boxes, box_labels=labels, topk=3):
    print(triplet)
# (person) --riding [0.76]--> (horse)
# (horse) --carrying [0.51]--> (person)
# (horse) --walking past [0.43]--> (car)

# 3. Change the relations to look for, without retraining.
model.set_vocabulary(["riding", "carrying a rider", "beside", "in front of"])
graphs = model.predict(image, boxes, box_labels=labels, decompose=True, topk=3)
print(graphs["spatial"])   # where things are, e.g. (horse) --in front of--> (car)
print(graphs["semantic"])  # what they do, e.g. (person) --riding--> (horse)