import sys
import urllib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

import boto3
import io

## ML STUFF ##
class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

def load_classifier_stuff(frontal_face_cascade_path='haarcascade_frontalface_default.xml', weights_path='./trained-models/facial-landmark-detection.pth'):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + frontal_face_cascade_path)  # Note the change
    best_network = Network()
    best_network.load_state_dict(torch.load(
        weights_path, map_location=torch.device('cpu')))
    best_network.eval()

    return face_cascade, best_network

def pre_process_image(image):
    display_image = image
    grayscale_image = image.convert('L')
    display_image = np.array(display_image)
    grayscale_image = np.array(grayscale_image)
    return display_image, grayscale_image

def get_preds(pre_processed_image, network, harr):
    faces = harr.detectMultiScale(pre_processed_image, 1.1, 4)

    all_landmarks = []

    for (x, y, w, h) in faces:
        image = pre_processed_image[y:y+h, x:x+w]
        image = TF.resize(Image.fromarray(image), size=(224, 224))
        image = TF.to_tensor(image)

        with torch.no_grad():
            landmarks = network(image.unsqueeze(0))

        landmarks = (landmarks.view(68, 2).detach().numpy()) * \
            np.array([[w, h]]) + np.array([[x, y]])
        all_landmarks.append(landmarks)

    return all_landmarks

def overlay_preds(display_image, all_landmarks, filename, relative_save_path):
    file_path = relative_save_path + '/' + filename
    plt.figure()
    plt.imshow(display_image)
    for landmarks in all_landmarks:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='c', s=5)
    plt.savefig(file_path)
    data = open(file_path, 'rb')
    return data, filename, file_path

## IMAGE DOWN/UP LOAD WITH S3 ##
def get_s3_image(bucket_name, key):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(key)
    file_stream = io.BytesIO()
    object.download_fileobj(file_stream)
    image = Image.open(file_stream)
    return image

def upload_processed_data_to_s3(data, key_name, bucket_name):
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).put_object(Key=key_name, Body=data)

## LAMBDA FUNCTION HANDLER ##
def handler(event, context):
    bucket_for_unprocessed_image = event["Records"][0]["s3"]["bucket"]["name"]
    bucket_for_processed_image = "facial-landmark-post"
    key_for_unprocessed_image = urllib.parse.unquote(
        event["Records"][0]["s3"]["object"]["key"], encoding="utf-8"
    )
    
    print(f'Getting image from bucket: {bucket_for_unprocessed_image}, with key: {key_for_unprocessed_image}')

    harr_class, trained_net = load_classifier_stuff()

    image = get_s3_image(bucket_for_unprocessed_image, key_for_unprocessed_image)

    display_image, pre_processed_image = pre_process_image(image)

    pred_landmarks = get_preds(pre_processed_image, trained_net, harr_class)

    data, filename, _ = overlay_preds(display_image, pred_landmarks, key_for_unprocessed_image, '/tmp')
    upload_processed_data_to_s3(data, filename, bucket_for_processed_image)
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('facial-landmark-post')
    table.put_item(
    Item={
            'uuid': filename
        }
    )

    return 'Image Processed' + sys.version + '!' + bucket_for_processed_image + " " + bucket_for_processed_image