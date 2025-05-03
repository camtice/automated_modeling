import numpy as np
import random
import math
from typing import Dict, List, Any
from scipy.optimize import minimize
import pandas as pd


def stable_logistic(utility, temperature=1):
    """Compute a stable logistic function to avoid overflow."""
    epsilon = 1e-10
    temperature = max(temperature, epsilon)
    utility_clipped = np.clip(utility / temperature, -100, 100)
    return 1 / (1 + np.exp(-utility_clipped))


def get_param_bounds(learnable_params: Dict) -> List[tuple]:
    """Convert parameter ranges to bounds format for scipy.optimize."""
    bounds = []
    for param_name, info in learnable_params.items():
        try:
            range_info = info.get("range", {})
            if not range_info:
                raise ValueError(
                    f"No range information found for parameter {param_name}"
                )

            min_val = float(range_info.get("min", float("-inf")))
            max_val = float(range_info.get("max", float("inf")))

            if min_val == float("-inf") or max_val == float("inf"):
                raise ValueError(f"Parameter {param_name} must have finite bounds")

            bounds.append((min_val, max_val))
        except Exception as e:
            raise ValueError(f"Error getting bounds for {param_name}: {str(e)}")

    if not bounds:
        raise ValueError("No valid parameter bounds found in learnable_params")

    return bounds


def random_starting_points(bounds: List[tuple], n_starts: int = 3) -> List[np.ndarray]:
    """Generate random starting points within bounds."""
    points = []
    for _ in range(n_starts):
        point = [random.uniform(lower, upper) for lower, upper in bounds]
        points.append(np.array(point))
    return points


def negative_log_likelihood_utility(
    params: np.ndarray,
    trial_data: List[Dict],
    simulation_code: str,
    param_names: List[str],
    target_value_name: str,
) -> float:
    """Calculate negative log likelihood for given parameters for binary choice models."""
    try:
        # Convert numpy values to Python floats
        params = [float(p) for p in params]

        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))

        # Create a local namespace for execution
        local_vars = {"math": math, "random": random}

        # Execute simulation code with current parameters
        exec(simulation_code, local_vars)
        result = local_vars["simulate_model"](trial_data, **param_dict)

        if not isinstance(result, list):
            raise ValueError(f"simulate_model returned {type(result)}, expected list")

        # Calculate log likelihood
        total_ll = 0
        for utility, trial in zip(result, trial_data):
            p_correct = stable_logistic(utility)
            actual_decision = trial[target_value_name]
            p = p_correct if actual_decision == 1 else (1 - p_correct)
            total_ll += np.log(p + 1e-10)

        return -total_ll

    except Exception as e:
        print(f"Error in specific operation: {e}")
        raise ValueError(
            f"Error in likelihood calculation: {str(e)}\nParams: {param_dict}"
        )


def mean_squared_error(
    params: np.ndarray,
    trial_data: List[Dict],
    simulation_code: str,
    param_names: List[str],
    target_value_name: str,
) -> float:
    """Calculate mean squared error for numerical value prediction models. Skips trials where target value is nan or None."""
    try:
        # Convert numpy values to Python floats
        params = [float(p) for p in params]

        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))

        # Create a local namespace for execution
        local_vars = {"math": math, "random": random}

        # Execute simulation code with current parameters
        exec(simulation_code, local_vars)
        predicted_values = local_vars["simulate_model"](trial_data, **param_dict)

        if not isinstance(predicted_values, list):
            raise ValueError(
                f"simulate_model returned {type(predicted_values)}, expected list"
            )

        # Calculate MSE
        total_squared_error = 0
        valid_trials = 0

        for predicted, trial in zip(predicted_values, trial_data):
            try:
                actual_value = trial[target_value_name]

                # Skip trials where target value is nan or None
                if actual_value is None or (
                    isinstance(actual_value, float) and math.isnan(actual_value)
                ):
                    continue

                actual_value = float(actual_value)
                squared_error = (predicted - actual_value) ** 2
                total_squared_error += squared_error
                valid_trials += 1
            except (TypeError, ValueError, KeyError):
                # Skip this trial if any errors occur
                continue

        # Check if we have any valid trials
        if valid_trials == 0:
            raise ValueError("No valid trials found for MSE calculation")

        mse = total_squared_error / valid_trials
        return mse

    except Exception as e:
        raise ValueError(f"Error in MSE calculation: {str(e)}\nParams: {param_dict}")


def calculate_accuracy(
    params: Dict[str, float],
    trial_data: List[Dict],
    simulation_code: str,
    target_value_name: str,
) -> float:
    """Calculate prediction accuracy for binary choice models by simulating decisions.

    Args:
        params: Dictionary of parameter names and values
        trial_data: List of dictionaries containing trial data
        simulation_code: Python code string for the simulation model
        target_value_name: Name of the target variable in the trial data

    Returns:
        float: Accuracy as proportion of correctly predicted decisions
    """
    try:
        # Create a local namespace for execution
        local_vars = {"math": math, "random": random}

        # Execute simulation code with current parameters
        exec(simulation_code, local_vars)
        utilities = local_vars["simulate_model"](trial_data, **params)

        if not isinstance(utilities, list):
            raise ValueError(
                f"simulate_model returned {type(utilities)}, expected list"
            )

        correct_predictions = 0
        total_predictions = 0

        # Simulate decisions based on model probabilities and compare to actual decisions
        for utility, trial in zip(utilities, trial_data):
            # Calculate probability from utility using stable logistic
            p_choose = stable_logistic(utility)

            # Simulate decision based on probability
            simulated_decision = 1 if random.random() < p_choose else 0

            # Get actual decision
            actual_decision = trial[target_value_name]

            # Count correct predictions
            if simulated_decision == actual_decision:
                correct_predictions += 1

            total_predictions += 1

        # Calculate accuracy
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        return accuracy

    except Exception as e:
        print(f"Error in accuracy calculation: {e}")
        return 0.0


def fit_participant(
    participant_data: List[Dict],
    simulation_code: str,
    learnable_params: Dict[str, Dict],
    target_value_name: str,
    prediction_type: str,
) -> Dict[str, Any]:
    """Fit parameters for a single participant.

    Args:
        participant_data: List of dictionaries containing trial data
        simulation_code: Python code string for the simulation model
        learnable_params: Dictionary of learnable parameters with their constraints
        target_value_name: Name of the target variable in the trial data
        prediction_type: Type of prediction ("utility" or "numerical")

    Returns:
        Dictionary containing fitted parameters and optimization results
    """
    try:
        bounds = get_param_bounds(learnable_params)
        param_names = list(learnable_params.keys())

        # Validate data
        if not participant_data:
            raise ValueError("No data provided for participant")

        if target_value_name not in participant_data[0]:
            raise ValueError(f"Missing '{target_value_name}' column in data")

        # Select the appropriate objective function based on prediction type
        if prediction_type.lower() == "utility":
            objective_function = negative_log_likelihood_utility
            optimization_goal = "likelihood"  # We maximize likelihood
        elif prediction_type.lower() == "numerical_variable_estimation":
            objective_function = mean_squared_error
            optimization_goal = "mse"  # We minimize MSE
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "args": (participant_data, simulation_code, param_names, target_value_name),
            "options": {
                "maxiter": 1000,
                "ftol": 1e-6,
            },
        }

        # Generate multiple random starting points
        initial_points = random_starting_points(bounds, n_starts=3)

        best_result = None
        best_objective = float("inf")  # For minimization (both -LL and MSE)

        optimization_errors = []

        for init_params in initial_points:
            try:
                # Test the objective function first
                test_objective = objective_function(
                    init_params,
                    participant_data,
                    simulation_code,
                    param_names,
                    target_value_name,
                )

                if not np.isfinite(test_objective):
                    raise ValueError(
                        f"Initial objective value is not finite: {test_objective}"
                    )

                result = minimize(objective_function, init_params, **minimizer_kwargs)

                # For likelihood, we negate the fun value to get the actual LL
                # For MSE, we use the fun value directly
                current_objective = result.fun
                if optimization_goal == "likelihood":
                    current_objective = (
                        -current_objective
                    )  # Convert back to LL from -LL

                # For likelihood, higher is better. For MSE, lower is better.
                # We're minimizing, so for likelihood we want to compare negated values
                is_better = False
                if optimization_goal == "likelihood":
                    is_better = (
                        current_objective > best_objective if best_result else True
                    )
                else:  # MSE
                    is_better = (
                        current_objective < best_objective if best_result else True
                    )

                if result.success and is_better:
                    best_result = result
                    best_objective = current_objective

            except Exception as e:
                optimization_errors.append(str(e))
                continue

        if best_result is None:
            raise ValueError(
                f"All optimization attempts failed. Errors: {optimization_errors}"
            )

        # Create results dictionary
        fit_results = {name: value for name, value in zip(param_names, best_result.x)}

        # Store appropriate metric in results
        if optimization_goal == "likelihood":
            # Calculate accuracy for utility models
            param_dict = {
                name: value for name, value in zip(param_names, best_result.x)
            }
            model_accuracy = calculate_accuracy(
                params=param_dict,
                trial_data=participant_data,
                simulation_code=simulation_code,
                target_value_name=target_value_name,
            )

            fit_results.update(
                {
                    "success": best_result.success,
                    "log_likelihood": -best_result.fun,  # Convert back to log likelihood from negative log likelihood
                    "optimization_message": best_result.message,
                    "prediction_type": "utility",
                    "accuracy": model_accuracy,  # Add accuracy to results
                }
            )
        else:  # MSE
            fit_results.update(
                {
                    "success": best_result.success,
                    "mse": best_result.fun,  # MSE is already in the correct format
                    "optimization_message": best_result.message,
                    "prediction_type": "numerical",
                }
            )

        # Log the results before returning
        print(f"Fit results for participant: {fit_results}") # Added logging

        return fit_results

    except Exception as e:
        # Add logging for the exception before raising it
        print(f"Error during fitting participant data: {str(e)}") # Log the error
        raise ValueError(f"Error in parameter fitting: {str(e)}")


def get_dataset_info(csv_path: str) -> Dict[str, Any]:
    """
    Extract column names and data types from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary containing column names and their data types
    """
    try:
        df = pd.read_csv(csv_path)
        # Get column names and their data types
        column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {
            "variables": list(df.columns),
            "data_types": column_info,
            "n_rows": len(df),
        }
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {"error": str(e)}
