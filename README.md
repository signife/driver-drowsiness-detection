# driver-drowsiness-detection
A deep learning project to detect driver drowsiness using computer webcam. Features real-time monitoring and alert system. 


## 📝 Overview
### Execution Order
- 1. Environment Setup
```bash
pip install -r requirements.txt
```
- using Dataset in roboflow. you can also downloading in here -> <a href="https://universe.roboflow.com/karthik-madhvan/drowsiness-detection-xsriz">Dataset Link</a>

- following below command prompt to proceed with training
```bash
yolo task=detect mode=train model=yolov8s.pt data={./data}/data.yaml epochs=100 imgsz=640
'''
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
CheerCharm_Front/
├── public/                    # 정적 파일(이미지, 폰트 등) 모음 폴더
├── src/                       # 소스 코드 포함 폴더
│   ├── components/            # 컴포넌트를 담고 있는 폴더
│   ├── pages/                 # 페이지 컴포넌트를 담고 있는 폴더
│   ├── styles/                # 전역적으로 사용되는 스타일 시트 관련 파일을 담고 있는 폴더
│   ├── api/                   # 재사용성이 높은 함수 등 유틸리티 함수를 담고 있는 폴더
│   ├── App.js                 # 라우팅, 상태 관리 등 전체적인 애플리케이션의 엔트리 포인트 파일
│   ├── index.js               # React DOM 렌더링을 위한 파일
├── .env                       # 환경 변수 파일
└── README.md                  # 리드미 파일
```
