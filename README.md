# Model Training Instructions

Follow the steps below to set up and run the model:

## 1. Install Python Environment
- Install Python (version 3.8+ recommended).  
- You can use an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/).


## 2. Download the Code
- Download the file **`model_trainingV2.py`** to your local computer.  

## 3. Download the Dataset
- The dataset consists of two folders:  
  - **`healthy`**  
  - **`fusarium`**  
- Place these two folders inside a new folder named **`train`** on your computer.  

Your folder structure should look like this:

<img width="223" height="319" alt="image" src="https://github.com/user-attachments/assets/84196890-4a59-4222-901f-440379fbaf20" />


## 4. Update Dataset Path in Code
- Open **`model_trainingV2.py`**.  
- Find the following line:
  ```python
  path = "C://train"

##  Replace it with the actual path to your train folder. Example:

path = "D://Projects/Dataset/train"

