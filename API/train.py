from  tm_common import *
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Embedding,
    Dense,
    Input,
    Embedding,
)

import mlflow
from pathlib import Path


def get_data_sets(output_dir): 
    
    _, _, y_train, y_test = tm_load_train_test_set(output_dir)

    X_corpus_train, X_corpus_test = tm_load_train_test_set(output_dir, option="raw_corpus")
    for i in range(3) : 
        X_corpus_train, y_train = tm_get_subset(X_corpus_train, y_train)
        X_corpus_test, y_test = tm_get_subset(X_corpus_test, y_test)

    y_train_b, y_test_b = tm_multilabel_binarizer(y_train, y_test)
    
    return X_corpus_train, X_corpus_test, y_train_b, y_test_b


    
def get_use_model() : 
    use_model = Sequential()
    use_model.add(Dense(256, activation='relu', input_dim=512))
    use_model.add(Dense(128, activation='relu'))
    use_model.add(Dense(30,activation='sigmoid'))
    # Compile the model
    use_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return use_model


def get_optimum_threshold(y_train_b, y_pred) : 
    
    threshold_test_use = tm_test_threshold(y_train_b[:100], y_pred) 

    max_config = threshold_test_use.query('precision == precision.max()')
    t = max_config["threshold"]
    best_t =  t.values[0]
    return  best_t

if __name__ == "__main__" : 
    output_dir = 'C:\dev\\topic_modelling\output\\'
    model_path = 'C:\dev\\topic_modelling\API\model\\tm_use.keras'
    BATCH_SIZE = 128
    EPOCH = 100
    exp_id = 1 
    print("[USE TRAIN] Load use embedding")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("[USE TRAIN] done")
    
    mlflow.autolog()

    experiment_id = mlflow.create_experiment(
        "NLP_TOPIC_MODELLING ",
        artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        tags={"version": "v1", "priority": "P1"},
    )
    experiment = mlflow.get_experiment(experiment_id)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))
    
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run : 

        print("[USE TRAIN] Active run_id: {}".format(run.info.run_id))
        
        mlflow.set_tracking_uri("file://" + os.path.join(output_dir,"logs\mlruns"))
        tracking_uri = mlflow.get_tracking_uri()
        print("[USE TRAIN] Current tracking uri: {}".format(tracking_uri))
        
        
        
        print("[USE TRAIN] Load data ...")
        X_corpus_train, X_corpus_test, y_train_b, y_test_b = get_data_sets(output_dir)
        print(len(X_corpus_train))
        print(len(X_corpus_test))
        print("[USE TRAIN] Load completed")
        
        print("[USE TRAIN] Generate the embedding of the data ...")
        X_train_use_embedding = embed(X_corpus_train.tolist())
        X_test_use_embedding = embed(X_corpus_test.tolist())
        print("[USE TRAIN] embedding completed")

    
        print("[USE TRAIN] Get model and fit ...")
        use_model = get_use_model()
        use_model.summary()
        use_model.fit(X_train_use_embedding, y_train_b, batch_size = 128, epochs=100)
        

        y_pred = use_model.predict(X_train_use_embedding[:100])
        best_t = get_optimum_threshold(y_train_b[:100], y_pred) 
        print("[USE TRAIN] Decision threshold : ",best_t)
        print("[USE TRAIN] Model trained")
        
        print("[USE TRAIN] Evaluation .... ")
        y_pred_use = use_model.predict(X_test_use_embedding)
        y_pred_use = (y_pred_use > best_t).astype(np.float32)
        
        precision_use      = average_precision_score(y_test_b, y_pred_use, average='micro')
        jaccard_score_use = jaccard_score(y_test_b, y_pred_use, average='micro')

        print("[USE TRAIN] done")
        
        print("[USE TRAIN] Save model .... ")
        use_model.save(model_path)
        
        print("[USE TRAIN] done")

    mlflow.delete_experiment(experiment_id)
