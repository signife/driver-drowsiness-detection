# driver-drowsiness-detection
A deep learning project to detect driver drowsiness using computer webcam. Features real-time monitoring and alert system. 


## 📝 Overview
### Execution Order


-**Environment Setup**
```bash
pip install -r requirements.txt
```
-**Data Preparation** 

Using Dataset in roboflow. you can also downloading in here -> <a href="https://universe.roboflow.com/karthik-madhvan/drowsiness-detection-xsriz">Dataset Link</a>


Following below command prompt to proceed with training
```bash
yolo task=detect mode=train model=yolov8s.pt data={./data}/data.yaml epochs=100 imgsz=640
```

-**Execution**


You can skip the first two steps by downloading the best.pt file that we previously uploaded.


Please execute the code within a virtual environment where the requirements have been installed.
```bash
python main.py
```



### 📄 References
- we followed the algorithm from the referenced paper from here -> <a href="https://www.ijeat.org/wp-content/uploads/papers/v8i5S/E10150585S19.pdf">Paper Link</a>
![image](https://github.com/signife/driver-drowsiness-detection/assets/87606589/e91af02b-27d9-42e4-a7bb-3ee9df2701da)




### 🛠️ Tech Stack
![Python Badge](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)
![Anaconda Badge](https://img.shields.io/badge/Anaconda-44A833?style=flat&logo=Anaconda&logoColor=white)
![OpenCV Badge](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white)
![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white)
![YOLOv8 Badge](https://img.shields.io/badge/YOLOv8-FFA500?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAiklEQVR42mL8//8/AzUBEQNGJkU4DDXwzGcNQFAY1QGECQNEHjwZL8oMwk6FBOAjDlC7HBhU/gZPogR4gyYhBLVcAOxlAaVwDQTsUM1UPgyYYBmVwMlVsUMU1YKJVBkR2EFkXgACkABYQ/wcAaWI3gGAV6ADcVJcAhDmRBBwDAACRMIK3v5Dh0AAAAASUVORK5CYII=&logoColor=white)


### 👥 Teammates

|  Name  |  github-links   |
|:----------:|:----------:|
|    ChangHo Jang    |  <a href="https://github.com/Chris99ChangHo">Chris99ChangHo</a>  |


### 📁 Directory Structure
```
driver-drowsiness-detection/
├── main.py/                   
├── alarm.wav/                      
├── best.pt/        
├── requirements.txt/                 
└── README.md                 
```
