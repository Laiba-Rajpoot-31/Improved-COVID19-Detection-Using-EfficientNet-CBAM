import numpy as np

def generate_cardinality(N, p=2):
    """
    Generates default fuzzy measures if none are provided.
    """
    return [(x / N)**p for x in np.arange(N, 0, -1)]

def sugeno_fuzzy_integral_generalized(X, measure, axis=0, f1=np.minimum, f2=np.amax, keepdims=False):
    """
    Computes the generalized Sugeno Fuzzy Integral.
    This preserves the exact mathematical formulation from the original paper.
    """
    # Sort the probability scores in ascending order
    X_sorted = np.sort(X, axis=axis)
    
    # Calculate the minimum between the sorted probabilities and the fuzzy measures, 
    # then take the maximum of those minimums.
    return f2(f1(np.take(X_sorted, np.arange(0, X_sorted.shape[axis]), axis), measure), axis=axis, keepdims=keepdims)

def ensemble_sugeno(prob_list, measures=None):
    """
    Combines the probability outputs of multiple models using the Sugeno Fuzzy Integral.
    
    Args:
        prob_list (list of np.ndarray): A list where each element is the probability 
                                        matrix (num_samples, num_classes) of a model.
        measures (list or np.ndarray): The fuzzy measures for the classifiers.
                                       Must be the same length as prob_list.
                                       
    Returns:
        predictions (np.ndarray): The final predicted class indices for each sample.
        Y (np.ndarray): The computed integral values for each class.
    """
    num_models = len(prob_list)
    num_samples = prob_list[0].shape[0]
    num_classes = prob_list[0].shape[1]
    
    if measures is None:
        # Using default measures matching the repository logic if not specified.
        # Ensure the length matches the number of models.
        if num_models == 4:
            measures = np.array([1.5, 1.5, 0.01, 1.2])
        else:
            measures = np.array(generate_cardinality(num_models))
    else:
        measures = np.array(measures)
        
    if len(measures) != num_models:
        raise ValueError(f"Number of measures ({len(measures)}) must match number of models ({num_models}).")
        
    Y = np.zeros((num_samples, num_classes), dtype=float)
    
    # Iterate through all samples and classes to calculate the integral
    for sample in range(num_samples):
        for cls in range(num_classes):
            # Gather the probabilities of all models for this specific sample and class
            X = np.array([prob_list[m][sample][cls] for m in range(num_models)])
            
            # Apply Sugeno Fuzzy Integral
            integral_val = sugeno_fuzzy_integral_generalized(X, measures)
            Y[sample][cls] = integral_val
            
    # Final prediction is the class that yields the maximum Sugeno Integral value
    predictions = np.argmax(Y, axis=1)
    
    return predictions, Y
