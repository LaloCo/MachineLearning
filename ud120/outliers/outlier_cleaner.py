#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    errors = []
    cleaned_errors = []

    ### your code goes here
    count = 0
    for p in predictions:
        errors.append((net_worths[count] - p) * (net_worths[count] - p))
        count = count + 1
    
    sorted_errors = sorted(errors)
    outlier_definer = sorted_errors[80]

    cleaned_net_worths = net_worths[errors < outlier_definer]
    cleaned_ages = ages[errors < outlier_definer]
    cleaned_predictions = predictions[errors < outlier_definer]

    print(len(cleaned_net_worths))

    count = 0
    for p in cleaned_predictions:
        cleaned_errors.append((cleaned_net_worths[count] - p) * (cleaned_net_worths[count] - p))
        count = count + 1

    cleaned_data = tuple(zip(cleaned_ages, cleaned_net_worths, cleaned_errors))
    
    return cleaned_data