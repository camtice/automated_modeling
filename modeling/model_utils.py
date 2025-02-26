import numpy as np
import random
import math
from typing import Dict, List, Any
from scipy.optimize import minimize
import pandas as pd


def stable_logistic(utility, temperature=2):
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


def negative_log_likelihood(
    params: np.ndarray,
    trial_data: List[Dict],
    simulation_code: str,
    param_names: List[str],
) -> float:
    """Calculate negative log likelihood for given parameters."""
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
            p_accept = stable_logistic(utility)
            actual_decision = trial["accept"]
            p = p_accept if actual_decision == 1 else (1 - p_accept)
            total_ll += np.log(p + 1e-10)

        return -total_ll

    except Exception as e:
        raise ValueError(
            f"Error in likelihood calculation: {str(e)}\nParams: {param_dict}"
        )


def fit_participant(
    participant_data: List[Dict],
    simulation_code: str,
    learnable_params: Dict[str, Dict],
) -> Dict[str, Any]:
    """Fit parameters for a single participant."""
    try:
        bounds = get_param_bounds(learnable_params)
        param_names = list(learnable_params.keys())

        # Validate data
        if not participant_data:
            raise ValueError("No data provided for participant")

        if "accept" not in participant_data[0]:
            raise ValueError("Missing 'accept' column in data")

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "args": (participant_data, simulation_code, param_names),
            "options": {
                "maxiter": 1000,
                "ftol": 1e-6,
            },
        }

        # Generate multiple random starting points
        initial_points = random_starting_points(bounds, n_starts=3)

        best_result = None
        best_likelihood = float("-inf")

        optimization_errors = []

        for init_params in initial_points:
            try:
                # Test the likelihood function first
                test_ll = negative_log_likelihood(
                    init_params, participant_data, simulation_code, param_names
                )

                if not np.isfinite(test_ll):
                    raise ValueError(f"Initial likelihood is not finite: {test_ll}")

                result = minimize(
                    negative_log_likelihood, init_params, **minimizer_kwargs
                )

                if result.success and -result.fun > best_likelihood:
                    best_result = result
                    best_likelihood = -result.fun

            except Exception as e:
                optimization_errors.append(str(e))
                continue

        if best_result is None:
            raise ValueError(
                f"All optimization attempts failed. Errors: {optimization_errors}"
            )

        # Create results dictionary
        fit_results = {name: value for name, value in zip(param_names, best_result.x)}

        fit_results.update(
            {
                "success": best_result.success,
                "log_likelihood": -best_result.fun,
                "optimization_message": best_result.message,
            }
        )

        return fit_results

    except Exception as e:
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
