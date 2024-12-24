import evaluate
import numpy as np
from sklearn import metrics
import torch
import numpy as np
import re

def classification_metrics(eval_pred):


    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    
    precision = metrics.precision_score(labels, predictions, average="macro")
    recall = metrics.recall_score(labels, predictions, average="macro")
    f1 = metrics.f1_score(labels, predictions, average="macro")
    accuracy = metrics.accuracy_score(labels, predictions)
    
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}



def process_predictions(predictions, tokenizer):
    """
    Decode predictions and extract numerical values.
    Args:
        predictions: A batch of token IDs (batch_size, seq_len).
        tokenizer: Tokenizer to decode token IDs into text.
    Returns:
        Array of numerical values extracted from the predictions.
    """
    predictions =  np.argmax(predictions, axis=-1)
    processed_predictions = []
    for token_ids in predictions:  # Iterate over the batch
        # Ensure token_ids is a list of integers
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()  # Convert tensor to list

        #print('token_ids', token_ids)
        
        # Decode the prediction (list of token IDs to string)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Extract numerical value using regex
        match = re.search(r'[-+]?\d*\.\d+|\d+', decoded_text)
        if match:
            value = float(match.group())  # Convert matched value to float
        else:
            value = 0.0  # Default value if no number is found
        processed_predictions.append(value)
    
    return np.array(processed_predictions)  # Return as numpy array


def regression_metrics(eval_pred):
    """
    Compute regression metrics for predictions and labels.
    """
    # Unpack predictions and labels
    predictions, labels = eval_pred
    #print('hi predictions', predictions.shape, type(predictions))  # (batch, seq, dimension)
    #print('hi labels', labels.shape)  # (batch,)

    # Decode predictions and convert to numerical values
    predictions = process_predictions(predictions, tokenizer)  # Shape: (batch,)
    
    # Ensure labels are numpy arrays for metrics calculation
    labels = np.array(labels)

    # Compute regression metrics
    mae = metrics.mean_absolute_error(labels, predictions)
    mse = metrics.mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = metrics.r2_score(labels, predictions)  # Coefficient of determination

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

