import evaluate
import numpy as np
from sklearn import metrics
import torch
import numpy as np
import re
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class ClassificationMetrics:
    def __call__(self, eval_pred):
        """
        Compute classification metrics for predictions and labels.
        Args:
            eval_pred: An EvalPrediction object containing (logits, labels).
        Returns:
            A dictionary containing precision, recall, F1-score, and accuracy.
        """
        # Extract logits and labels properly
        logits, labels = eval_pred.predictions, eval_pred.label_ids  

        # If logits is a tuple, take the first element
        if isinstance(logits, tuple):
            logits = logits[0]  

        # Convert logits to a NumPy array
        logits = np.array(logits)  

        # Ensure logits has the correct shape before applying argmax
        predictions = np.argmax(logits, axis=-1)

        # Compute metrics
        precision = metrics.precision_score(labels, predictions, average="macro")
        recall = metrics.recall_score(labels, predictions, average="macro")
        f1 = metrics.f1_score(labels, predictions, average="macro")
        accuracy = metrics.accuracy_score(labels, predictions)

        return {"precision": precision, "recall": recall, "f1-score": f1, "accuracy": accuracy}


class RegressionMetrics:
    def __init__(self, tokenizer=None, error_boundary = 0.001):
        self.tokenizer = tokenizer
        self.error_boundary=error_boundary

    def process_predictions(self, predictions, tokenizer):
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

            #print('decoded_text', decoded_text)
            
            # Extract numerical value using regex
            matches = re.findall(r'[-+]?\d*\.\d+|\d+', decoded_text)
            #print('match', match)
            if matches:
                value = float(matches[-1]) # Convert matched value to float
            else:
                value = 0.0  # Default value if no number is found
            processed_predictions.append(value)
        
        return np.array(processed_predictions)  # Return as numpy array
    
    def __call__(self, eval_pred):
        predictions, labels = eval_pred

        # Decode predictions and convert to numerical values
        if self.tokenizer is not None:
            predictions = self.process_predictions(predictions, self.tokenizer)  # Shape: (batch,)
        
        # Ensure labels are numpy arrays for metrics calculation
        labels = np.array(labels)

        # Compute regression metrics
        mae = metrics.mean_absolute_error(labels, predictions)
        mse = metrics.mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = metrics.r2_score(labels, predictions)  # Coefficient of determination
        corr = 0
        incorr = 0
        for i, j in zip(labels, predictions):
        
            if abs(i-j) < self.error_boundary:
                corr += 1
            else:
                incorr += 1
        acc = corr/(corr+incorr)
                
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Accuracy":acc,
            "R2": r2,
            "Pearson" : pearsonr(predictions,labels)[0],
            "Spearman's Rank":spearmanr(predictions,labels)[0]
            
        }
