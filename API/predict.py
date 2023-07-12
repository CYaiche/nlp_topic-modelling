import os 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
from tensorflow import keras

class TopicModellingModel : 
    def __init__(self):
        dir_name = os.getcwd()
        threshold_path = os.path.join(dir_name,'model/best_t.npy')
        label_list = os.path.join(dir_name,'model/label_list.npy')
        model_path = os.path.join(dir_name,'model/tm_use.keras')

        assert (
            os.path.exists(threshold_path)
            and os.path.exists(label_list)
            and os.path.exists(model_path)
        ), "model files not existent"


        use_encoder = "https://tfhub.dev/google/universal-sentence-encoder/4"
        print("[USE inference] Load use embedding")
        self.embed = hub.load(use_encoder)
        print("[USE INFERENCE] done")

        print("[USE INFERENCE] Load classification model")
        self.model = keras.models.load_model(model_path)
        print("[USE INFERENCE] done")

        print("[USE INFERENCE] Load threshold and label")
        self.best_t = np.load(threshold_path)
        self.labels = np.load(label_list, allow_pickle=True)
        print("[USE INFERENCE] done")
        
    def run_inference(self,input_string) : 
        
        print("[USE INFERENCE] Run inference")
        print("[USE INFERENCE] Run embedding")
        embedding = self.embed([input_string])
        print("[USE INFERENCE] Run classification")
        y_pred = self.model.predict(embedding)
        print("[USE INFERENCE] Run threshold")
        output_classification = (y_pred > self.best_t).astype(np.float32)[0]
        print("[USE INFERENCE] inference done")
        
        print("[USE INFERENCE] Run id to tag")

        output_labels = [
            self.labels[i]
            for i in range(len(output_classification))
            if output_classification[i] > 0
        ]
        print("[USE INFERENCE] done")
        
        return output_labels

if __name__ == "__main__" : 
    print("[USE INFERENCE] run local tests")
    use_topic_model = TopicModellingModel()
    
    f = open('C:\dev\\topic_modelling\API\\tests\c_posts.txt','r')
    contents = f.read()
    output = use_topic_model.run_inference(contents)
    print(output)
    
    f = open('C:\dev\\topic_modelling\API\\tests\json_posts.txt','r')
    contents = f.read()
    output = use_topic_model.run_inference(contents)
    print(output)
        
    f = open('C:\dev\\topic_modelling\API\\tests\pandas_python.txt','r')
    contents = f.read()
    output = use_topic_model.run_inference(contents)
    print(output)
    