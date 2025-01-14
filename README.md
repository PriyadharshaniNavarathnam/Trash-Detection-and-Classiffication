# Trash Classification and Detection
## Contibuter
- Dhishanth P.
- Priyadharshani N.

Recycling trash is a crucial part of protecting our environment. Garbage must be divided into categories with similar recycling processes in order to enable the recycling process. The percentage of recycled waste can rise considerably if it is possible to separate domestic trash into several categories. Using the classes given, we trained the model in this notebook to categorize the input images and output the trash classification.
Garbage classification and detection aim to automatically identify and categorize waste into predefined classes, such as plastic, metal, paper, organic, and hazardous waste.
This project aims to classify trash into various categories using deep learning techniques. The model leverages MobileNetV2 for efficient and accurate image classification.

---

## Applications
- Automated waste segregation at recycling plants.
- Real-time litter detection in public spaces.
- Smart garbage bins with embedded classification capabilities.
- Environmental monitoring for reducing waste pollution.

## Dataset

The Garbage Classification Dataset contains 6 classifications: cardboard (393), glass (491), metal (400), paper(584), plastic (472), and trash(127).
TrashNet: A popular dataset containing images of waste categorized into classes like paper, glass, metal, plastic, and cardboard.
TACO (Trash Annotation in Context): A large dataset with annotated images of litter in real-world scenarios.
The dataset can be found on the [Garbage Classification](https://www.kaggle.com/asdasdasasdas/garbage-classification) page on Kaggle, or downloaded directly through [here](https://www.kaggle.com/asdasdasasdas/garbage-classification/download).

TACO is a growing image dataset of waste in the wild. It contains images of litter taken under diverse environments: woods, roads and beaches. These images are manually labeled and segmented according to a hierarchical taxonomy to train and evaluate object detection algorithms.
The dataset currently contain 60 different classes.
For convenience, annotations are provided in COCO format.
## Module Overview

In this module I built and trained a neural network to classify different recyclable objects using PyTorch.<br>
I based my program on the [Garbage Classification Dataset](https://www.kaggle.com/asdasdasasdas/garbage-classification).


# Trash Classification

## Features

- **Multi-Class Classification**: Detects trash categories like paper, plastic, metal, glass, etc.
- **Custom Dataset Support**: Easily adaptable to other datasets.


## Training the Model
- Open trash_classification.ipynb in Jupyter Notebook or Google Colab.
- Update the dataset path and training parameters as needed.
- Execute the notebook to train the classification model.

## MobileNetV2 Architecture
- ![image](https://github.com/user-attachments/assets/cb301135-4cd9-4192-ad7e-5b8d72a131ab)


## Testing the Model
- Use the Streamlit interface (app/app.py) for real-time image classification.
- Alternatively, run the testing cells in trash_classification.ipynb for batch testing.

## Visualizations
![image](https://github.com/user-attachments/assets/eeaf6bbf-3201-4ba5-b548-ca467f00064d)

![image](https://github.com/user-attachments/assets/fd6cf103-57cd-4bff-ab04-47ff6376247c)

![image](https://github.com/user-attachments/assets/3f8a606c-ce03-46a5-a992-6622080a01d8)

![image](https://github.com/user-attachments/assets/c1d1a676-c2e2-4cbc-b49e-1d09da202cd5)

![image](https://github.com/user-attachments/assets/e0f503f5-5421-47f3-b813-306f39c0d166)

![image](https://github.com/user-attachments/assets/84e67e39-37a5-4935-86ae-000d59217ad1)
![image](https://github.com/user-attachments/assets/edb7ef94-9c7b-4847-a32b-886c21693f7b)
![image](https://github.com/user-attachments/assets/f19b1df8-813c-43ad-9f79-2a0ace8c2be6)
![image](https://github.com/user-attachments/assets/c7081abb-9984-457f-88c0-529aa1edd6a1)
![image](https://github.com/user-attachments/assets/0b2106d7-75f0-4929-82c4-cc95e38da534)

![image](https://github.com/user-attachments/assets/8593999a-fe87-4079-9ad0-71dd94da6092)

## Requirements
- Python Libraries
      - TensorFlow
      - NumPy
      - Pandas
      - Matplotlib
       - Streamlit
      - Scikit-learn


  ## Future Enhancements
- Implement object detection for trash in images.
- Use transfer learning for better performance on smaller datasets.
- Extend the dataset to include more trash categories.
- Implement streamlit app


  
# Trash Detection



## Overview
This project focuses on detecting trash in images using a deep learning approach. The core model used appears to be MobileNet, tailored to identify various types of trash effectively. The project is likely part of a hackathon or similar challenge, as indicated by the file naming and operations performed within the notebook.

## Project Structure
The Jupyter notebook, `Trash_Detection.ipynb`, contains the complete workflow for training the model, evaluating its performance, and saving the trained model to Google Drive. The notebook is structured to run in Google Colab, leveraging its GPU capabilities for training.

### Contents of the Notebook
- Data loading and preprocessing
- Model setup using MobileNet architecture
- Training the model with validation
- Visualizing training results
- Saving the trained model to Google Drive

## Model Information
- Architecture: MobileNet
- Training Data: Assumed to be a collection of labeled images containing various types of trash.
- Output: Model capable of detecting and classifying trash in images.


## Setup and Running Instructions
1. **Open Google Colab**: Start by opening Google Colab in your browser.
2. **Upload the Notebook**: Upload the `Trash_Detection.ipynb` file to Colab.
3. **Mount Google Drive**: To save and access files directly, mount your Google Drive in the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
## Visualisations
![image](https://github.com/user-attachments/assets/e03bf522-03de-42f1-aa4c-66b9b88cca09)

![image](https://github.com/user-attachments/assets/b8ecd930-bc40-4930-bc2f-35bc87fe2b57)
![image](https://github.com/user-attachments/assets/4402deb0-2554-4625-a603-220e8d14e49e)
![image](https://github.com/user-attachments/assets/17b9f2d7-1612-4ac3-bbdc-11c5a6507f3b)

![image](https://github.com/user-attachments/assets/c11ffaf0-7f5a-4dc7-8664-197ea16cb782)
![image](https://github.com/user-attachments/assets/10e96dfd-7879-4e2c-98d4-70c95ce41d76)

![image](https://github.com/user-attachments/assets/aa994f16-53bf-43f2-b1ec-7aed01f73322)


## Future Work
- Enhancements: Potential improvements could include refining the model with more data, implementing better preprocessing techniques, or experimenting with different model architectures.
- Deployment: Steps towards deploying the model in a real-world application or integrating with a web or mobile application.





### Notes:
- **Assumptions**: Some assumptions were made about the contents of the notebook based on common practices and snippets observed. Please adjust or extend sections as necessary based on the actual content and specific details of your project.
- **Document Details**: Remember to include any specific installation requirements, additional setup steps, or data handling instructions as needed.
- **Results and Evaluation**: Add specific sections detailing performance metrics, charts, or any key findings from the experiments conducted in the notebook.












  
