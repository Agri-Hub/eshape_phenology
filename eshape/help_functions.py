import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

def phenology_extraction_for_one_feature(ref_values, end_days, data, phenology_scale_range, feature, diff_days=5,
                                         slack=3, metric="mae", stat='derivative', bestN=3, tw=75, round_to=1, 
                                         validation = None, verbose = 1):
    '''
    :param ref_values: DataFrame, The feature space of the reference parcels
    :param end_days: list, Iterable, Series or set. It contains the Days of Prediction (DoP)
    :param data: DataFrame, The feature space of the examined parcels
    :param phenology_scale_range: Dictionary, Contains the doy ranges of each phenological stage
    :param feature: String, The feature to be used for the phenology calculation
    :param diff_days: int, default = 5, Number of days to calculate the derivatives
    :param slack: int, default = 1, Number of days to slide during the comparisons
    :param metric: "mse", "mae" or "dtw", default = "mae"
    :param stat: "amplitude" or "derivative", default = "derivative"
    :param bestN: int, default = 3, The amount of the stages with the smallest errors to return
    :param tw: int, default = 75, Days prior to the Day of Prediction
    :param round_to: int, default = 1, if > 1 results are rounded to the nearest muliple of round_to
    :param validation: DataFrame, default None. Contains the validation information. If provided, the input of the 
           end_days argument is ignored and values are acquired from the corresponing column of the Dataframe
    :param verbose: int
    Controls the verbosity: the higher, the more messages.
        >0 : information about each prediction are printed
        >1 : plot of the signatures is also printed


    :return: Dictionary of dictionaries, for each parcel and for each DoP the predicted stages

    '''

    dict_phenology = {}
    keys = ref_values.unique_id.unique()
    for i, parcel in enumerate(data.id):
        before_stage = 100
        dict_phenology[parcel] = {}
        min_mse = 10000000
        
        if validation is not None:
            end_days = validation.loc[validation.id == parcel , "doy"]
        
        for end_doy in end_days:
            tw = min(tw, end_doy - 110)
            start_doy = end_doy - tw
            cols = [x for x in data.columns[1:] if start_doy <= int(x[-3:]) < end_doy]
            cols = [x for x in cols if feature in x]
            fs5day = data.loc[i, cols]
            if stat == 'derivative':
                fs5day = [np.arctan((fs5day[i + diff_days - 1] - fs5day[i]) / diff_days) for i in
                          range(0, len(cols) - diff_days + 1)]
            errors = []
            for b, (st, end) in phenology_scale_range.items():
                if end_doy < st or end_doy > end:
                    continue
                for key in keys:
                    startNP = ref_values.loc[(ref_values.unique_id == key) & (ref_values.stage == b), "doy"].values[0]
                    endNP = ref_values.loc[(ref_values.unique_id == key) & (ref_values.stage == b + 100), "doy"].values[0]
                    for j in range(startNP, endNP, 3):
                        cols_ref = [v for v in ref_values.columns[5:] if j - tw < int(v[-3:]) <= j]
                        cols_ref = [x for x in cols_ref if feature in x]
                        ref5 = ref_values.loc[(ref_values.unique_id == key) & (ref_values.stage == b), cols_ref].values[0]
                        if stat == 'derivative':
                            ref5 = [np.arctan((ref5[i + diff_days - 1] - ref5[i]) / diff_days) for i in
                                    range(0, len(ref5) - diff_days + 1)]
                        if len(ref5) != len(fs5day):
                            continue
                        w = [0.1 * (i + 1) for i in range(len(fs5day))]
                        if metric == "mae":
                            error = mae(fs5day, ref5, sample_weight=w)
                        elif metric == "mse":
                            error = mse(fs5day, ref5, sample_weight=w)
                        elif metric == 'dtw':
                            matches, error, mapping_1, mapping_2, matrix = dtw(fs5day, ref5)
                        else:
                            print("Error on metric attribute. Metric can be one of 'mse', 'mae' or 'dtw'")
                        if error < min_mse:
                            min_mse = error
                        ref5_2 = ref_values.loc[(ref_values.unique_id == key) & (ref_values.stage == b), cols_ref].values[0]
                        pos_temp = (j - startNP) / (endNP - startNP)
                        errors.append((error, b, pos_temp, endNP - startNP, key, j, ref5, ref5_2))

            errors = sorted(errors, key=lambda tup: tup[0])

            my_scale = [round_to * round((round(x[2] * 100) + x[1]) / round_to) for x in errors]

            if verbose > 0:
                print("--------------------------------------")
                print("{} for parcel {} at day {}.".format(feature, parcel, end_doy))
                print("Predictions: {}".format(my_scale[:3]))
                if validation is not None:
                      stage = validation.loc[(validation.id == parcel) & (validation.doy == end_doy), "stage"].values[0]
                      print("True label: {}".format(stage))      

            if verbose > 1:
                plt.title("{} for parcel {}".format(feature, parcel))
                plt.plot(np.arange(tw), data.loc[i, cols], label='Input parcel')
                plt.plot(np.arange(tw), errors[0][-1],
                         label='best_error1_{:.0f}'.format(my_scale[0]))
                plt.plot(np.arange(tw), errors[1][-1],
                         label='best_error2_{:.0f}'.format(my_scale[1]))
                plt.plot(np.arange(tw), errors[2][-1],
                         label='best_error3_{:.0f}'.format(my_scale[2]))
                plt.legend()
                plt.show()
            dict_phenology[parcel][end_doy] = (my_scale[:3], [x[4] for x in errors[:3]])
            before_stage = final_stage
#             print()
    return dict_phenology


def final_stage(row):
    '''
    :param row: Series, series of stages for different features
    :return: int, final predicted stage
    '''

    values = np.array([x[0] for x in row.values])
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(values, mean_value, std_value)
    values = values[(values > mean_value - 2.5 * std_value) & (values < mean_value + 2.5 * std_value)]
    stage = int(np.sort(values)[len(values) // 2])
    return stage


def most_common_ref_parcel(row, stage):
    '''
    :param row: Series, series of stages for different features
    :param stage: int, final phenological stage
    :return: string, the id of the most dominant parcel
    '''


    d = {}
    for name, value in zip(row.index, row.values):
        if (int(value[0]) == stage):
            d[value[1]] = d.setdefault(value[1], 0) + 1
    return max(d, key=d.get)


def mae_to_doys(stage, pred_stage, durations):
    '''
    :param stage: int, final phenological stage
    :param pred_stage: int, predicted phenological stage
    :param durations: dictionary, duration of each phenological stage
    :return:
    '''
    ph1, ph2 = 100 * int(stage // 100), 100 * int(pred_stage // 100)
    if ph1 == ph2:
        return abs(pred_stage - stage) * (durations[ph1]) / 100
    elif ph1 < ph2:
        c = abs(ph1 + 100 - stage) * (durations[ph1]) / 100
        for i in range(ph1 + 100, ph2 + 100, 100):
            if i == ph2:
                if i == 700:
                    break
                c += abs(pred_stage - ph2) * (durations[i]) / 100
            else:
                c += durations[i]
        return c
    else:
        c = abs(ph2 + 100 - stage) * (durations[ph2]) / 100
        for i in range(ph2 + 100, ph1 + 100, 100):
            if i == ph1:
                if i == 700:
                    break
                c += abs(pred_stage - ph1) * (durations[i]) / 100
            else:
                c += durations[i]
        return c

def phenology_extraction(ref_values, end_days, data, phenology_scale_range, features, diff_days=5,
                         slack=3, metric="mae", stat='derivative', bestN=3, tw=75, round_to=1, 
                         validation = None, verbose = 1):
    results = {}
    for feature in features:
        stages = phenology_extraction_for_one_feature(ref_values, end_days, data, phenology_scale_range, feature,
                        metric = metric, stat = stat, bestN = bestN,  tw=tw, round_to=round_to,
                        validation = validation, verbose = verbose)
        results[feature] = stages
    return results

