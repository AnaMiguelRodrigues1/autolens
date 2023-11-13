import time
import os
import random

from fastai.vision.all import *
from fastai.test_utils import *
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from fastai.callback.tracker import CSVLogger
from src.utils import handle_dataset
from src.utils.create_resources_folder import resources

def main(path_metadata: str,
        path_dataset: str,
        steps: int,
        target_size: tuple,
        test_size: float, 
        valid_size: float
        ):

    """
    :param path_metadata:
    :param path_dataset:
    :param steps:
    :param test_size:
    :param valid_size:
    :param target_size:
    :return:
    """

    resources()

    print('Time --> Start')
    start_time = time.time()

    print('Loading Data')
    dataset = handle_dataset.check(path_dataset)
    n_data = dataset.n_data
    train, test, valid = dataset.to_path(test_seed=random.randint(1, 10000), 
                                        val_seed=random.randint(1, 10000),
                                        test_size=test_size,
                                        valid_size=valid_size)

    # Converting data to a format compatible to Ktrain
    train['is_valid'] = False
    valid['is_valid'] = True
    train_valid = [train, valid]
    df = pd.concat(train_valid)

    print('Creating Dataloaders')
    dls = ImageDataLoaders.from_df(
            df,
            path='',
            label_col='binary_label', # multiclass_label (according to column in metadata)
            valid_col='is_valid',
            num_workers=5, #5
            bs=16, # batch size
            item_tfms=Resize(256),
            batch_tfms=aug_transforms()
            )

    print('Importing Pre-Trained Model')
    # Model Examples: densenet201, inceptionV3, resnet50
    learn = vision_learner(
            dls, 
            resnet34,  
            metrics=[error_rate, accuracy, F1Score(average='weighted')], # average='macro'
            ps=0, 
            wd=0
            )

    # Add CSVLogger to log to save the history in a csv file
    csv_logger = CSVLogger(fname='resources/fastai_ghost.csv')
    learn.add_cb(csv_logger)
     
    learn.recorder.valid_metrics = True
    learn.recorder.train_metrics = True

    print('... layers still frozen')

    # Find appropriate lr
    lr_min_1, lr_steep_1, lr_slide_1, lr_valley_1 = learn.lr_find(suggest_funcs=(minimum, steep, slide, valley))
    print('lr_valley:', lr_valley_1)

    learn.fit_one_cycle(20, lr_valley_1) #20
    
    print('... unfreezing the layers')
    learn.unfreeze()

    print('Fitting Model')
    Find once again the appropriate lr
    lr_min_2, lr_steep_2, lr_slide_2, lr_valley_2 = learn.lr_find(suggest_funcs=(minimum, steep, slide, valley))
    print('lr_min', lr_min_2)

    learn.fit_one_cycle(15, lr_min_2) #10 epochs for the other two

    # Model is not save automatically
    # learn.save('path/to/model')

    histories = {}
    csv_path = 'resources/fastai_ghost.csv'
    df = pd.read_csv(csv_path)

    for column in df.columns:
        histories[column] = [df[column].tolist()]

    # Fastai does not provide learn.evaluate()

    print('Predictions')
    tst_dl = dls.test_dl(test)
    probs,target,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)

    # Predicted labels
    y_pred = idxs.tolist()

    # Probabilities of the predicted labels
    #y_prob = [prob for prob in probs.tolist()]
    y_prob = np.array([prob[1] for prob in probs.tolist()])

    # Actual labels
    y_true = test['multiclass_label'].tolist()

    predictions_and_more = {}
    predictions_and_more['y_pred']=[y_pred]
    predictions_and_more['y_prob']=[y_prob]
    predictions_and_more['y_true']=[y_true]

    # Accuracy, Precision, Recall, F1-Score, AUC, MCC
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    scores={}
    results=[acc, prec, rec, f1, mcc, auc]
    metric_names=['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']
    for i in range(len(results)):
        rounded_values = round(results[i], 4)
        scores[metric_names[i]] = rounded_values

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    scores['time']=elapsed_time
    print('Time:', elapsed_time)

    print('Saving Results...')
    df_1 = pd.DataFrame(histories)
    df_2 = pd.DataFrame(scores, index=[0])
    df_3 = pd.DataFrame(predictions_and_more)
    all_results = pd.concat([df_1, df_2, df_3], axis=1)

    all_results.to_csv('resources/fastai_results.csv', mode='a', index=False)
    print('Results Ready!')

    

    


