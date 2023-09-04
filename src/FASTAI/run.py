import time
import os

from fastai.vision.all import *
from fastai.test_utils import *
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from fastai.callback.tracker import CSVLogger
from src.utils.handle_hist import load_dataset
from src.utils.handle_fastai_lrs import find_lr_range


def main(path_metadata: str,
        path_dataset: str,
        steps: int,
        target_size=(255, 255)):
    """
    :param path_metadata:
    :param path_dataset:
    :param steps:
    :param target_size:
    :return:
    """

    print('Time --> Start')
    start_time = time.time()

    dataset = load_dataset(path_metadata, path_dataset)
    n_data = dataset.n_data
    train, test, valid = dataset.to_path(test_seed=random.randint(1, 10000), val_seed=random.randint(1, 10000))
    #train, test, valid = dataset.to_path()

    histories = {}
    scores = {}
    predictions_and_more = {}

    print('Preparing Data')
    train['is_valid'] = False
    valid['is_valid'] = True
    train_valid = [train, valid]
    df = pd.concat(train_valid)

    dls = ImageDataLoaders.from_df(
            df,
            path='',
            label_col='binary_label',
            valid_col='is_valid', 
            num_workers=0, 
            bs=12,
            item_tfms=Resize(460),
            batch_tfms=aug_transforms()
            )

    print('Importing Pre-Trained Model')
    # Model Examples: densenet201, inceptionV3, vgg16
    learn = vision_learner(
            dls, 
            resnet50, 
            metrics=[error_rate, accuracy, F1Score(average='binary')],
            ps=0, 
            wd=0)

    # Add CSVLogger to log training metrics to a CSV file
    csv_logger = CSVLogger(fname='resources/fastai_ghost.csv')
    learn.add_cb(csv_logger)
     
    learn.recorder.valid_metrics = True
    learn.recorder.train_metrics = True

    print('... layers still frozen')
    # Find appropriate lr
    lr_min_1, lr_steep_1, lr_slide_1, lr_valley_1 = learn.lr_find(suggest_funcs=(minimum, steep, slide, valley))
    print('while frozen, lr_valley:', lr_valley_1)

    learn.fit_one_cycle(1, lr_valley_1) #1
    
    print('... unfreezing the layers')
    learn.unfreeze()

    print('Fitting Model')
    lr_min_2, lr_steep_2, lr_slide_2, lr_valley_2 = learn.lr_find(suggest_funcs=(minimum, steep, slide, valley))
    print('while unfrozen, lr_min', lr_min_2)

    learn.fit_one_cycle(10, lr_min_2) #10

    print('Saving Model')
    learn.save('vgg16')

    print('Saving Learning Curves')
    csv_path = 'resources/fastai_ghost.csv'
    df = pd.read_csv(csv_path)

    for column in df.columns:
        histories[column] = [df[column].tolist()]

    print('Evaluating Model')
    tst_dl = dls.test_dl(test)
    probs,target,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)

    y_pred = idxs.tolist()
    y_prob = [prob[1] for prob in probs.tolist()]
    y_prob = np.array(y_prob)
    y_true = test['binary_label'].tolist()
    
    # Accuracy, Precision, Recall, F1-Score, AUC, MCC
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    results=[acc, prec, rec, f1, auc, mcc]
    metric_names=['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
    for i in range(len(results)):
        rounded_values = round(results[i], 4)
        scores[metric_names[i]] = rounded_values

    print('Scores', scores)

    print('Predictions')
    predictions_and_more['y_pred']=[y_pred]
    predictions_and_more['y_prob']=[y_prob]
    predictions_and_more['y_true']=[y_true]

    print('Time --> Stop')
    elapsed_time = time.time() - start_time
    scores['time']=elapsed_time
    print('TIME', elapsed_time)

    print('Saving Results')
    
    df_1 = pd.DataFrame(histories) # maybe problem will arise here
    df_2 = pd.DataFrame(scores, index=[0])
    df_3 = pd.DataFrame(predictions_and_more)
    all_results = pd.concat([df_1, df_2, df_3], axis=1)

    all_results.to_csv('resources/fastai_results.csv', mode='a', index=False)
    print('Results Ready!')

    

    


