# Crack-Detection

- Developed, tested, and integrated machine learning models for detecting cracks in WRB and Seam Taping using PyTorch, improving defect detection capabilities.
- Designed and implemented data preprocessing pipelines, and created scalable backend services using FastAPI, Docker, and microservice architecture to support the automated detection system.


The purpose of this project is to compare the performances of 3 approaches (Azure Custom Vision, Faster R-CNN, and YOLO) on the given two datasets (SeamTaping and WRB).

## Preparation

### Prepare Python environment on local machine for this project.

Install **Miniconda**.

Then, launch **Anaconda Prompt (miniconda3)**.

```Anaconda
(base) C:\Users\User>D:
(base) D:\>cd Work
(base) D:\Work>conda create -n Crack-Detection python
(base) D:\Work>conda activate Crack-Detection
(Crack-Detection) D:\Work>pip install scikit-learn pandas openpyxl matplotlib pillow tqdm requests azure-cognitiveservices-vision-customvision python-dotenv
```

### Create an AI Services - Custom Vision resource on Azure and get the keys, ids, and endpoints.

Create an AI Services - Custom Vision resource on Azure starting from here: [Home - Microsoft Azure](https://portal.azure.com/#home)

Get the keys, ids, and endpoints: [Custom Vision - Settings](https://www.customvision.ai/projects#/settings)

Save into [Environment File](<different-approaches/azure-custom-vision/.env>)

Example:
```
TRAINING_ENDPOINT=https://crackdetection.cognitiveservices.azure.com/
TRAINING_KEY=90dad624b6664556accbcfd69e2e170d
PREDICTION_ENDPOINT=https://crackdetection-prediction.cognitiveservices.azure.com/
PREDICTION_KEY=4db3cee628434f8a9b492b9760036505
PREDICTION_RESOURCE_ID=/subscriptions/ddb01653-a592-4bb7-89e5-b39c2fc6e697/resourceGroups/Crack-Dtection/providers/Microsoft.CognitiveServices/accounts/CrackDetection-Prediction
```

## Check and Update the Two Datasets

There are 2 original datasets given in XLSX files:

- [WRB Bad Detection](dataset/WRB_All.xlsx)
- [SeamTaping Damaged Detection](dataset/SeamTaping_All.xlsx)

Please run the 5 notebooks in the subfolder '/dataset_check_update' before trying 3 approaches.

## Try Three Approaches

- Azure Custom Vision

    * Dataset injection to Azure Custom Vision
        + [SeamTaping Dataset Injection](<different-approaches/azure-custom-vision/1. inject-dataset__SeamTaping.ipynb>)
        + [WRB Dataset Injection](<different-approaches/azure-custom-vision/1. inject-dataset__WRB.ipynb>)

    * Train Model
        + [SeamTaping Model Training](<different-approaches/azure-custom-vision/2. train-model__SeamTaping.ipynb>)
        + [WRB Model Training](<different-approaches/azure-custom-vision/2. train-model__WRB.ipynb>)

    * Evaluate Performance
        + [SeamTaping Model Evaluation](<different-approaches/azure-custom-vision/3. evaluate-model__SeamTaping.ipynb>)
        + [WRB Model Evaluation](<different-approaches/azure-custom-vision/3. evaluate-model__WRB.ipynb>)

- Fasteer R-CNN

    * Prepare Dataset
        + [For WRB](<different-approaches/faster-rcnn/1. prepare-dataset_colab__WRB.ipynb>)

    * Train Faster R-CNN
        + [For WRB](<different-approaches/faster-rcnn/2. train-FasterRCNN_colab__WRB.ipynb>)

    * Evaluate Faster R-CNN
        + [For WRB](<different-approaches/faster-rcnn/3. evaluate-FasterRCNN_colab__WRB.ipynb>)


- YOLO v10
