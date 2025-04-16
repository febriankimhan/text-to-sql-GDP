"""Evaluation module for text-to-SQL results."""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from modules.constants import ColumnMatrix, ColumnName

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Class to store evaluation results for a single query.

    Args:
        index (int): Index of the query
        precision (float): Precision score
        recall (float): Recall score
        f1_score (float): F1 score
        accuracy (float): Accuracy score
        predicted_but_not_in_ground_truth (List[Dict[str, Any]]): False positives - items predicted but not in ground truth
        ground_truth_but_not_predicted (List[Dict[str, Any]]): False negatives - items in ground truth but not predicted
    """
    index: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    predicted_but_not_in_ground_truth: List[Dict[str, Any]]
    ground_truth_but_not_predicted: List[Dict[str, Any]]

class TextToSQLEvaluator:
    """Class to evaluate text-to-SQL results by comparing predictions with expected results."""

    def __init__(self, pred_results: List[List[Dict[str, Any]]], exp_results: List[List[Dict[str, Any]]]):
        """Initialize the evaluator with prediction and expected results.

        Args:
            pred_results (List[List[Dict[str, Any]]]): List of predicted query results
            exp_results (List[List[Dict[str, Any]]]): List of expected query results
        """
        self.pred_results = self.transform_keys(pred_results)
        self.exp_results = self.transform_keys(exp_results)
        self.evaluation_results: List[EvaluationResult] = []

    @staticmethod
    def transform_keys(data_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Transform dictionary keys from snake_case to Title Case format.

        Args:
            data_list (list[list[dict]]): Nested list of dictionaries with snake_case keys

        Returns:
            list[list[dict]]: Transformed nested list of dictionaries with Title Case keys
        """
        def transform_dict(d):
            # Create a new dictionary with transformed keys
            return {
                ' '.join(word.capitalize() for word in k.split('_')): v
                for k, v in d.items()
            }

        # Transform each dictionary in the nested structure
        # Handle None values by replacing them with empty lists
        return [
            [transform_dict(item) for item in (sublist or [])]
            for sublist in data_list
        ]

    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary keys and values for comparison.

        Args:
            d (Dict[str, Any]): Dictionary to normalize

        Returns:
            Dict[str, Any]: Normalized dictionary with lowercase keys and standardized values
        """
        # Create a new normalized dictionary
        normalized = {}

        for k, v in d.items():
            # Normalize key to lowercase and remove spaces
            norm_key = k.lower().replace(' ', '_')

            # Convert numeric strings to float for consistent comparison
            if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').isdigit()):
                norm_value = float(v)
            else:
                norm_value = str(v).lower()

            normalized[norm_key] = norm_value

        return normalized

    def _convert_dict_to_hashable(self, d: Dict[str, Any]) -> Tuple:
        """Convert dictionary to hashable format for comparison.

        Args:
            d (Dict[str, Any]): Dictionary to convert

        Returns:
            Tuple: Hashable representation of normalized dictionary
        """
        # First normalize the dictionary
        normalized = self._normalize_dict(d)
        # Convert to sorted tuple of items for consistent comparison
        return tuple(sorted(normalized.items()))

    def _calculate_metrics(self, pred_result: List[Dict[str, Any]], exp_result: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """Calculate precision, recall, F1 score and accuracy based on data existence.

        Args:
            pred_result (List[Dict[str, Any]]): List of predicted items
            exp_result (List[Dict[str, Any]]): List of expected items

        Returns:
            Tuple[float, float, float, float]: Precision, recall, F1 score, and accuracy
        """
        # Handle empty list cases
        if not pred_result and not exp_result:
            # Both lists are empty - perfect match
            return 1, 1, 1, 1
        elif not pred_result or not exp_result:
            # One list is empty while other isn't - complete mismatch
            return 0, 0, 0, 0

        # Regular case - convert dictionaries to sets of hashable tuples for comparison
        pred_set = {self._convert_dict_to_hashable(d) for d in pred_result}
        exp_set = {self._convert_dict_to_hashable(d) for d in exp_result}

        # Calculate intersection
        correct_predictions = len(pred_set.intersection(exp_set))
        total_predictions = len(pred_set)
        total_expected = len(exp_set)

        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        recall = correct_predictions / total_expected if total_expected > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_predictions / max(total_predictions, total_expected) if max(total_predictions, total_expected) > 0 else 0
        if accuracy > 0.5:
            accuracy = 1
        else:
            accuracy = 0
        return precision, recall, f1, accuracy

    def _find_wrong_predictions(self, pred_result: List[Dict[str, Any]], exp_result: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Find false positives and false negatives in predictions compared to ground truth.

        Args:
            pred_result (List[Dict[str, Any]]): List of predicted items
            exp_result (List[Dict[str, Any]]): List of ground truth items (expected results)

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
                - predicted_but_not_in_ground_truth: False positives (items predicted but not in ground truth)
                - ground_truth_but_not_predicted: False negatives (items in ground truth but not predicted)
        """
        # Convert to sets using normalized hashable format
        pred_dict_map = {self._convert_dict_to_hashable(d): d for d in pred_result}
        exp_dict_map = {self._convert_dict_to_hashable(d): d for d in exp_result}

        pred_set = set(pred_dict_map.keys())
        exp_set = set(exp_dict_map.keys())

        # Find differences
        predicted_but_not_in_ground_truth = [pred_dict_map[item] for item in pred_set - exp_set]  # False positives
        ground_truth_but_not_predicted = [exp_dict_map[item] for item in exp_set - pred_set]      # False negatives

        return predicted_but_not_in_ground_truth, ground_truth_but_not_predicted

    def evaluate(self) -> List[EvaluationResult]:
        """Evaluate all queries and store results.

        Returns:
            List[EvaluationResult]: List of evaluation results for each query
        """
        self.evaluation_results = []

        for idx, (pred_result, exp_result) in enumerate(zip(self.pred_results, self.exp_results)):
            # Calculate metrics based on data existence
            precision, recall, f1, accuracy = self._calculate_metrics(pred_result, exp_result)

            # Find wrong predictions
            predicted_but_not_in_ground_truth, ground_truth_but_not_predicted = self._find_wrong_predictions(pred_result, exp_result)

            # Store results
            result = EvaluationResult(
                index=idx,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                predicted_but_not_in_ground_truth=predicted_but_not_in_ground_truth,
                ground_truth_but_not_predicted=ground_truth_but_not_predicted
            )
            self.evaluation_results.append(result)

        return self.evaluation_results

    def create_evaluation_dataframe(self, df_data_test: pd.DataFrame, list_generated_sql_query: List[str], EXP_ID: str) -> pd.DataFrame:
        """Create a pandas DataFrame containing detailed evaluation results.

        Args:
            df_data_test (pd.DataFrame): Original dataframe containing test data
            list_generated_sql_query (List[str]): List of generated SQL queries

        Returns:
            pd.DataFrame: DataFrame containing evaluation details
        """
        if not self.evaluation_results:
            self.evaluate()

        # Create list of dictionaries for DataFrame
        evaluation_data = []

        for idx, result in enumerate(self.evaluation_results):
            row_data = {
                ColumnMatrix.EXP_ID: EXP_ID,
                ColumnName.NO: idx + 1,
                ColumnName.PROMPT: df_data_test.iloc[idx][ColumnName.PROMPT],
                ColumnName.EXPECTED_SQL_QUERY: df_data_test.iloc[idx][ColumnName.EXPECTED_SQL_QUERY],
                ColumnName.GENERATED_SQL_QUERY: list_generated_sql_query[idx],
                ColumnName.GENERATED_QUERY_RESULT: df_data_test.iloc[idx][ColumnName.GENERATED_QUERY_RESULT],
                ColumnName.EXPECTED_QUERY_RESULT: self.exp_results[idx],
                ColumnMatrix.GROUND_TRUTH_BUT_NOT_PREDICTED: result.ground_truth_but_not_predicted,
                ColumnMatrix.PREDICTED_BUT_NOT_IN_GROUND_TRUTH: result.predicted_but_not_in_ground_truth,
                ColumnMatrix.ACCURACY: result.accuracy,
                ColumnMatrix.TIME_TAKEN: df_data_test.iloc[idx][ColumnMatrix.TIME_TAKEN],
                ColumnName.DATABASE_TYPE: df_data_test.iloc[idx][ColumnName.DATABASE_TYPE],
            }
            evaluation_data.append(row_data)

        # Create DataFrame
        return pd.DataFrame(evaluation_data)

    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all queries.

        Returns:
            Dict[str, float]: Dictionary containing average precision, recall, F1 score and accuracy
        """
        if not self.evaluation_results:
            self.evaluate()

        metrics = defaultdict(list)
        for result in self.evaluation_results:
            metrics[ColumnMatrix.PRECISION.lower()].append(result.precision)
            metrics[ColumnMatrix.RECALL.lower()].append(result.recall)
            metrics[ColumnMatrix.F1_SCORE.lower()].append(result.f1_score)
            metrics[ColumnMatrix.ACCURACY.lower()].append(float(result.accuracy))

        return {
            ColumnMatrix.AVG_PRECISION: float(np.mean(metrics[ColumnMatrix.PRECISION.lower()])),
            ColumnMatrix.AVG_RECALL: float(np.mean(metrics[ColumnMatrix.RECALL.lower()])),
            ColumnMatrix.AVG_F1_SCORE: float(np.mean(metrics[ColumnMatrix.F1_SCORE.lower()])),
            ColumnMatrix.AVG_ACCURACY: float(np.mean(metrics[ColumnMatrix.ACCURACY.lower()]))
        }

    def print_evaluation_summary(self) -> None:
        """Print a summary of the evaluation results.

        Prints average precision, recall, F1 score and accuracy metrics.
        Also shows detailed results for each query.
        """
        if not self.evaluation_results:
            self.evaluate()

        avg_metrics = self.get_average_metrics()
        print("\nOverall Metrics:")
        print(f"Average Precision: {avg_metrics[ColumnMatrix.AVG_PRECISION]:.4f}")
        print(f"Average Recall: {avg_metrics[ColumnMatrix.AVG_RECALL]:.4f}")
        print(f"Average F1 Score: {avg_metrics[ColumnMatrix.AVG_F1_SCORE]:.4f}")
        print(f"Average Accuracy: {avg_metrics[ColumnMatrix.AVG_ACCURACY]:.4f}")

        print("\nDetailed Results:")
        for result in self.evaluation_results:
            print(f"\nQuery {result.index+1}:")
            print(f"{ColumnMatrix.PRECISION}: {result.precision:.4f}")
            print(f"{ColumnMatrix.RECALL}: {result.recall:.4f}")
            print(f"{ColumnMatrix.F1_SCORE}: {result.f1_score:.4f}")
            print(f"{ColumnMatrix.ACCURACY}: {result.accuracy:.4f}")

            if result.predicted_but_not_in_ground_truth:
                print(f"\nItems in prediction but not in expected ({ColumnMatrix.PREDICTED_BUT_NOT_IN_GROUND_TRUTH}):")
                for item in result.predicted_but_not_in_ground_truth:
                    print(f"  {item}")

            if result.ground_truth_but_not_predicted:
                print(f"\nItems in expected but not in prediction ({ColumnMatrix.GROUND_TRUTH_BUT_NOT_PREDICTED}):")
                for item in result.ground_truth_but_not_predicted:
                    print(f"  {item}")