import pandas as pd

def save_results(histories, scores, predictions, prob_predictions, y_test, elapsed_time):
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()

    print('... training & validation data')

    if type(histories) is dict:
        if len(histories) == 2 and next(iter(histories))=='train': # ludwig
            learning_curves = {}
                                            

            ## Recorda-te daqui que o binary_label depende do numero labels -> criar def
            for key, value in histories['train']['label'].items(): #binary_label
                new_key = 'train_' + key    
                learning_curves[new_key] = [value]
            
            for key, value in histories['valid']['label'].items():
                new_key = 'valid_' + key
                learning_curves[new_key] = [value]

            df_1 = pd.DataFrame(learning_curves)
        else:
            df_1 = pd.DataFrame(histories, index=[0]) # autogluon
        
    else: # autokeras
        learning_curves = {}

        for history in histories:
            for metric_name, metric_values in history.history.items():
                if metric_name not in learning_curves:
                    learning_curves[metric_name] = [metric_values]

        learning_curves['epochs'] = [history.epoch]
        df_1 = pd.DataFrame(learning_curves)

    print('... evaluation data')
    if type(scores) is dict:
        if len(scores) == 2: # ludwig
            scores_new = {}
            for key, value in scores['multiclass_label'].items():
                scores_new[key]=value
            
            scores_new['time'] = elapsed_time
            df_2 = pd.DataFrame(scores_new, index=[0])
        
        else: # autogluon
            scores['time'] = elapsed_time
            df_2 = pd.DataFrame(scores, index=[0])
    
    else: # autokeras
        performance_metrics = {}

        metric_names = list(history.history.keys())
        for metric_name, score_value in zip(metric_names, scores[0]):
            performance_metrics[metric_name] = score_value

        performance_metrics['time'] = elapsed_time
        df_2 = pd.DataFrame(performance_metrics, index=[0])

    print('... predictions data')
    # autokeras, ludwig and autogluon   
    targets_and_predictions = {
        'y_test': [y_test], 
        'y_predic': [predictions],
        'y_prob': [prob_predictions]
        }

    df_3 = pd.DataFrame(targets_and_predictions)

    all_results = pd.concat([df_1, df_2, df_3], axis=1)

    return all_results
