import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
from tensorflow import keras


def inference(input_string) : 
    threshold_path = 'C:\dev\\topic_modelling\API\model\\best_t.npy'
    label_list = 'C:\dev\\topic_modelling\API\model\label_list.npy'
    model_path = 'C:\dev\\topic_modelling\API\model\\tm_use.keras'
    
    use_encoder = "https://tfhub.dev/google/universal-sentence-encoder/4"
    print("[USE TRAIN] Load use embedding")
    embed = hub.load(use_encoder)
    print("[USE TRAIN] done")
    
    print("[USE TRAIN] Load classification model")
    model = keras.models.load_model(model_path)
    print("[USE TRAIN] done")
    
    print("[USE TRAIN] Load threshold and label")
    best_t = np.load(threshold_path)
    labels = np.load(label_list, allow_pickle=True)
    print("[USE TRAIN] done")
    
    
    
    print("[USE TRAIN] Run inference")
    print("[USE TRAIN] Run embedding")
    embedding = embed([input_string])
    print("[USE TRAIN] Run classification")
    y_pred = model.predict(embedding)
    print("[USE TRAIN] Run threshold")
    output_classification = (y_pred > best_t).astype(np.float32)[0]
    print("[USE TRAIN] inference done")
    
    print("[USE TRAIN] Run id to tag")

    output_labels = [
        labels[i]
        for i in range(len(output_classification))
        if output_classification[i] > 0
    ]
    print("[USE TRAIN] done")
    
    return output_labels

if __name__ == "__main__" : 

    
    f = open('C:\dev\\topic_modelling\API\\tests\c_posts.txt','r')
    contents = f.read()
    output = inference(contents)
    print(output)
    
    f = open('C:\dev\\topic_modelling\API\\tests\json_posts.txt','r')
    contents = f.read()
    output = inference(contents)
    print(output)
        
    f = open('C:\dev\\topic_modelling\API\\tests\pandas_python.txt','r')
    contents = f.read()
    output = inference(contents)
    print(output)
    