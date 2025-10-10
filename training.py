from ultralytics import YOLO

model=YOLO("yolo11n.pt")  
model.train(data="dataset_custom.yaml",epochs=100,batch=16,imgsz=640,device="0",workers=0)
