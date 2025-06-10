## Current status

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    metric,
    Metric,
    Score,
    Scorer,
    scorer,
    accuracy,
    stderr,
    Target,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import Generate, TaskState, solver, generate, system_message
from inspect_ai.util import ExecResult, sandbox
import pandas as pd
from typing import Dict, Any, List
import json
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from model_utils import fit_participant, stable_logistic, get_dataset_info
import statistics
import yaml
from pathlib import Path
from inspect_ai.model import GenerateConfig
import time
import re
import logging

# importing solvers
# from solvers.model_design import model_design_solver


def load_config(config_path=None) -> dict:
    """Load configuration from YAML file with fallback to default config."""
    # Get the directory containing this script
    base_dir = Path(__file__).parent

    # Use provided config path or fall back to default
    if not config_path:
        config_path = base_dir / "configs" / "default_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Store the actual config path used for later copying
    config["_config_path"] = str(config_path)

    # Load prompt texts
    prompts_dir = base_dir / "configs" / "prompts"
    with open(prompts_dir / "task_description.txt", "r") as f:
        task_description = f.read().strip()

    with open(prompts_dir / "model_output_description.txt", "r") as f:
        config["model_output_description"] = f.read().strip()

    with open(prompts_dir / "instructions.txt", "r") as f:
        config["instructions"] = f.read().strip()

    # Get dataset information
    dataset_info = ""
    info = get_dataset_info(config["paths"]["dataset"])
    if "error" not in info:
        dataset_info = "\nDataset Structure:\n"
        dataset_info += "Variables available:\n"
        for var, dtype in info["data_types"].items():
            dataset_info += f"- {var} ({dtype})\n"
        dataset_info += f"\nNumber of observations: {info['n_rows']}"

    # Combine task description with dataset info
    config["task_description"] = f"{task_description}\n\nDataset:\n{dataset_info}"

    return config


# Load configuration at module level
CONFIG = load_config()

# At some pont I should split the task description and the request to create
TASK_DESCRIPTION = CONFIG["task_description"]

MODEL_OUTPUT_DESCRIPTION = CONFIG["model_output_description"]


@solver
def model_design_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Use CONFIG instead of hardcoded values
        model_name = CONFIG["model"]["name"]
        model_config = GenerateConfig(
            max_tokens=CONFIG["model"].get("max_tokens"),
        )
        custom_model = get_model(model_name, config=model_config)

        # Get parameter recovery information if available
        param_recovery = state.metadata.get("parameter_recovery", {})
        correlations = param_recovery.get("correlations", {})

        # Format previous models with tags before generating
        previous_models = ""

        # Format previous models
        for i, model in enumerate(state.metadata.get("previous_models", []), 1):
            previous_models += (
                f"\nModel {i}:\n<previous_model_{i}>\n{model}\n</previous_model_{i}>"
            )

        # Get prediction type from config
        prediction_type = CONFIG["comp_model_specifications"]["type"]
        prediction_type_info = ""
        if prediction_type.lower() == "utility":
            prediction_type_info = """
Your model should predict the utility of a binary choice. The utility will be converted to a probability using a logistic function with temperature 1, and then used to predict binary decisions.
"""
        elif prediction_type.lower() == "numerical_variable_estimation":
            prediction_type_info = """
Your model should directly predict a numerical value (not a binary choice). The model's predictions will be compared to actual values using mean squared error.
"""
        else:
            raise ValueError(f"Invalid prediction type: {prediction_type}")

        # Combine the task description with desired output description

        output_description = state.metadata["output_description"]

        current_instructions = state.metadata.get("instructions")
        if not current_instructions:
            logging.error("Instructions not found in state metadata!")
            raise ValueError("Instructions not found in state metadata!")

        # Check if output_description already contains "Previous Models:"
        if "Previous Models:" in output_description:
            # If it does, don't add previous_models again as they're already included
            prompt = f"""
Task Description: {state.metadata["task_description"]}

Desired Output Specification: {output_description}

Model Type: {prediction_type_info}

Instructions: {current_instructions}

Please think through this step by step, then provide your model specification and variable descriptions.
""".strip()
        else:
            # If not, add previous_models as before
            prompt = f"""
Task Description: {state.metadata["task_description"]}

Desired Output Specification: {output_description}

Model Type: {prediction_type_info}

Instructions: {current_instructions}

Previous Models:{previous_models if previous_models else "Not Provided"}

Please think through this step by step, then provide your model specification and variable descriptions.
""".strip()

        # Update the user prompt in state
        state.user_prompt.text = prompt

        # Generate using custom model
        output = await custom_model.generate(state.messages)
        state.output = output

        state.metadata.setdefault("complete_model_interaction", []).append(
            {
                "solver": "model_design_solver",
                "input": state.user_prompt.text,
                "output": output.completion,
            }
        )
        state.messages.append(output.message)

        # Extract model
        model_match = re.search(
            r"<MODEL>(.*?)</MODEL>", state.output.completion, re.DOTALL
        )
        model = model_match.group(1).strip() if model_match else ""

        # Extract variable descriptions
        vars_match = re.search(
            r"<VARIABLES>(.*?)</VARIABLES>", state.output.completion, re.DOTALL
        )
        var_text = vars_match.group(1).strip() if vars_match else ""

        # Extract summary
        summary_match = re.search(
            r"<SUMMARY>(.*?)</SUMMARY>", state.output.completion, re.DOTALL
        )
        summary = summary_match.group(1).strip() if summary_match else ""

        # Extract target variable
        target_var_match = re.search(
            r"<target_variable>(.*?)</target_variable>",
            state.output.completion,
            re.DOTALL,
        )
        target_variable = (
            target_var_match.group(1).strip() if target_var_match else None
        )

        # Raise error if no target variable is found
        if not target_variable:
            raise ValueError("No target variable specified in model description")

        # Parse variable descriptions as JSON and ensure consistent format
        var_dict = {}
        if var_text:
            try:
                var_dict = json.loads(var_text)
                # Ensure consistent format for each variable
                formatted_vars = {}
                for var_name, var_info in var_dict.get("variables", {}).items():
                    formatted_vars[var_name] = {
                        "description": var_info.get("description", ""),
                        "range": {
                            "min": var_info.get("range", {}).get("min", "-inf"),
                            "max": var_info.get("range", {}).get("max", "inf"),
                            "inclusive_min": var_info.get("range", {}).get(
                                "inclusive_min", True
                            ),
                            "inclusive_max": var_info.get("range", {}).get(
                                "inclusive_max", True
                            ),
                        },
                        "distribution": var_info.get("distribution", None),
                        "learnable": var_info.get("learnable", False),
                        "source": var_info.get("source", "calculated"),
                    }
                var_dict = {"variables": formatted_vars}

                # Extract and store learnable parameter names
                learnable_params = {
                    name: info
                    for name, info in formatted_vars.items()
                    if info.get("learnable", False)
                }
                state.metadata["learnable_parameters"] = learnable_params

            except json.JSONDecodeError:
                state.metadata["parsing_error"] = (
                    "Failed to parse variable descriptions JSON"
                )
                var_dict = {"variables": {}}

        # Store everything in metadata
        state.metadata["model_specification"] = model
        state.metadata["variable_descriptions"] = var_dict["variables"]
        state.metadata["model_summary"] = summary
        state.metadata["target_variable"] = target_variable  # Store the target variable
        state.metadata["full_reasoning"] = state.output.completion
        state.metadata["prediction_type"] = prediction_type  # Store the prediction type

        # Store the raw model (without tags) in metadata for final_model_summary_solver to use
        current_model = f"""Specification: {model}
Summary: {summary}
Target Variable: {target_variable}
Prediction Type: {prediction_type}"""
        state.metadata["current_model"] = current_model

        # We're not adding BIC or parameter recovery info here anymore
        # That will be handled by final_model_summary_solver which runs after
        # parameter_recovery_solver and bic_solver

        return state

    return solve


@solver
def simulate_model_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        max_retries = CONFIG["fitting"]["max_retries"]
        retry_count = 0

        # Initialize simulation_errors list in metadata if it doesn't exist
        state.metadata.setdefault("simulation_errors", [])

        while retry_count < max_retries:
            try:
                # Extract model, variables, and dataset info from previous state
                model_spec = state.metadata.get("model_specification")
                var_desc = state.metadata.get("variable_descriptions", {})
                dataset_info = state.metadata.get("dataset_info")
                learnable_params = state.metadata.get("learnable_parameters", {})

                # Track current retry attempt
                state.metadata["current_retry"] = retry_count

                # Format variable descriptions for the prompt
                var_desc_formatted = json.dumps({"variables": var_desc}, indent=2)

                # Update the prompt to use the formatted variable descriptions
                prompt = f"""
                Write Python code to simulate the following mathematical model using only Python standard library (no pandas or numpy).
                The code should take a list of dictionaries as input (each dictionary representing a trial) and return a value for each trial.

                IMPORTANT: Use these exact parameter names for learnable parameters: {list(learnable_params.keys())}

                Please write a function called `simulate_model` that:
                1. Takes a list of dictionaries as input (each dictionary contains trial data)
                2. Has parameter names that matches the variable descriptions exactly
                3. Implements the mathematical model above using only standard Python math operations
                4. Returns a list of model_predictions for each trial (please use the exact name "model_predictions" as the return value)

                Your code will be implemented within the following code block (please do not include any code outside of this code block, for instance, you should not include data_json = # imported from .json dumps, or data = json.loads(data_json)):

                ```python
import json
import math

[YOUR SIMULATE_MODEL FUNCTION HERE]

# Data as list of dictionaries
data_json = # imported from .json dumps
data = json.loads(data_json)

# Get results for the data
results = simulate_model(data)
print(json.dumps({{"results": results}}))  # Output as JSON for reliable parsing
'''

Below is an example for you to follow:
<EXAMPLE>
Model Specification:
U_i = β + εE + ηN

Variable Descriptions:
U_i: Utility (probability) of choosing action i β: [learnable] Base tendency parameter (0-1) representing inherent preference ε: [learnable] Environmental sensitivity parameter (0-1) E: Environmental cue value (-1 to 1) η: Noise parameter (0-1) N: Random noise drawn from normal distribution N(0,1)

Dataset Structure:
Variables available:
- trial_id (int64)
- environmental_cue (float64)
- beta (float64)
- epsilon (float64)
- eta (float64)
- utility (float64)
- choice_made (int64)
- trial_number (int64)
- participant_id (int64)


<YOUR RESPONSE>
def simulate_model(trial_data, beta=1, epsilon=1, eta=1):
    model_predictions = []
    for trial in trial_data:
        E = trial.get("environmental_cue", 0)
        u1, u2 = random.random(), random.random()
        N = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        model_predictions.append(beta + (epsilon * E) + (eta * N))
    return model_predictions
</YOUR RESPONSE>

Now, create a simulate_model function for the following model incorporating the information provided. DO NOT OUTPUT ANYTHING AFTER YOU RETURN THE MODEL PREDICTIONS.
                Model Specification:
                {model_spec}

                Variable Descriptions:
                {var_desc_formatted}

Dataset Structure:
{dataset_info}

                """

                state.user_prompt.text = prompt
                current_prompt = state.user_prompt.text
                state = await generate(state)
                # Log the complete interaction for debugging
                state.metadata.setdefault("complete_model_interaction", []).append(
                    {
                        "solver": "simulate_model_solver",
                        "input": current_prompt,
                        "output": state.output.completion,
                        "timestamp": time.time(),
                    }
                )

                # Clean up the generated code
                simulation_code = state.output.completion

                # Remove any Markdown code block delimiters
                simulation_code = simulation_code.replace("```python", "").replace(
                    "```", ""
                )

                # Extract the complete function definition including return statement
                function_match = re.search(
                    r"(def\s+simulate_model\s*\(.*?\):.*?return\s+model_predictions)",
                    simulation_code,
                    re.DOTALL,
                )

                if function_match:
                    # Use only the matched function definition, not the entire code block
                    simulation_code = function_match.group(1).strip()
                else:
                    error_msg = (
                        "Could not extract simulate_model function from generated code"
                    )
                    state.metadata["simulation_error"] = error_msg
                    state.metadata["simulation_errors"].append(
                        {
                            "retry_number": retry_count,
                            "error": error_msg,
                            "timestamp": time.time(),
                        }
                    )
                    retry_count += 1
                    continue  # Skip to next retry without raising an exception

                # Rest of the code to test the simulation...
                try:
                    # First read and parse the CSV in the main process
                    import pandas as pd
                    import math

                    df = pd.read_csv(CONFIG["paths"]["dataset"])
                    first_participant = df.iloc[0]["ID"]
                    participant_data = df[df["ID"] == first_participant]

                    # Convert participant data to list of dictionaries
                    data_dicts = participant_data.to_dict("records")

                    # Function to replace NaN with None
                    def replace_nan(obj):
                        if isinstance(obj, float) and math.isnan(obj):
                            return None
                        elif isinstance(obj, dict):
                            return {k: replace_nan(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [replace_nan(item) for item in obj]
                        else:
                            return obj

                    data_dicts_clean = replace_nan(data_dicts)

                    # Add default parameters for all learnable parameters
                    default_params = ", ".join(
                        [f"{param}=0.5" for param in learnable_params.keys()]
                    )

                    # Create test code with explicit parameters and clean data
                    test_code = f"""
import json
import math
import random

{simulation_code}

# Data as list of dictionaries
data = json.loads('''{json.dumps(data_dicts_clean)}''')

# Get results for the data
results = simulate_model(data, {default_params})
print(json.dumps({{"results": results}}))
"""

                    # Execute the test code
                    result = await sandbox().exec(
                        cmd=["python", "-c", test_code],
                        timeout=30,
                    )

                    if result.success:
                        try:
                            # Strip any extra whitespace and parse only the first line
                            output_line = result.stdout.strip().split("\n")[0]
                            output_data = json.loads(output_line)

                            if "error" in output_data:
                                state.metadata["simulation_error"] = output_data[
                                    "error"
                                ]
                                # Add error to simulation_errors list
                                state.metadata["simulation_errors"].append(
                                    {
                                        "retry_number": retry_count,
                                        "error": output_data["error"],
                                        "timestamp": time.time(),
                                    }
                                )
                                state.output.completion = (
                                    f"Error in simulation: {output_data['error']}"
                                )
                                # Don't raise an exception, just increment retry counter and continue
                                retry_count += 1
                                continue
                            else:
                                # Success! Store the results and return
                                simulation_results = output_data["results"]
                                state.metadata["simulation_code"] = simulation_code
                                state.output.completion = str(simulation_results)
                                # Record total retries on success
                                state.metadata["total_retries"] = retry_count
                                return state
                        except json.JSONDecodeError as e:
                            error_msg = (
                                f"JSON parsing error: {str(e)}\nOutput: {result.stdout}"
                            )
                            state.metadata["simulation_error"] = error_msg
                            # Add error to simulation_errors list
                            state.metadata["simulation_errors"].append(
                                {
                                    "retry_number": retry_count,
                                    "error": error_msg,
                                    "timestamp": time.time(),
                                }
                            )
                            # Don't raise an exception, just increment retry counter and continue
                            retry_count += 1
                            continue
                    else:
                        error_msg = result.stderr
                        state.metadata["simulation_error"] = error_msg
                        # Add error to simulation_errors list
                        state.metadata["simulation_errors"].append(
                            {
                                "retry_number": retry_count,
                                "error": error_msg,
                                "timestamp": time.time(),
                            }
                        )
                        state.metadata["simulation_code"] = simulation_code
                        state.output.completion = (
                            f"Error in simulation: {result.stderr}"
                        )
                        # Don't raise an exception, just increment retry counter and continue
                        retry_count += 1
                        continue

                except Exception as e:
                    error_msg = str(e)
                    state.metadata["simulation_error"] = error_msg
                    state.metadata["simulation_errors"].append(
                        {
                            "retry_number": retry_count,
                            "error": error_msg,
                            "timestamp": time.time(),
                        }
                    )
                    retry_count += 1
                    continue  # Continue to next retry without raising

            except Exception as e:
                # This catches any exceptions in the outer try block
                retry_count += 1
                error_msg = str(e)

                # Record error information
                state.metadata["simulation_errors"].append(
                    {
                        "retry_number": retry_count,
                        "error": error_msg,
                        "timestamp": time.time(),
                    }
                )

                if retry_count >= max_retries:
                    state.metadata["simulation_error"] = (
                        f"Failed after {max_retries} attempts: {error_msg}"
                    )
                    state.metadata["total_retries"] = retry_count
                    state.output.completion = f"Error in simulation: {error_msg}"
                    return state

                # Continue to next retry without raising
                continue

        # If we get here, we've exhausted all retries
        state.metadata["total_retries"] = retry_count
        state.metadata["simulation_error"] = "Failed after maximum retries"
        state.output.completion = (
            "Error: Failed to generate working simulation after maximum retries"
        )
        return state

    return solve


@solver
def parameter_fitting_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get necessary data from state
        simulation_code = state.metadata.get("simulation_code")
        var_desc = state.metadata.get("variable_descriptions", {})
        target_variable = state.metadata.get("target_variable")

        # Update: Use config for dataset path
        df = pd.read_csv(CONFIG["paths"]["dataset"])

        all_results = []
        participant_ids = df["ID"].unique()

        # Get prediction type from config
        prediction_type = CONFIG["comp_model_specifications"]["type"]

        # Track skipped participants
        skipped_participants = []

        # Fit for each participant
        for participant_id in participant_ids:
            try:
                participant_data = df[df["ID"] == participant_id]
                data_dicts = participant_data.to_dict("records")

                # Extract group information using the configured column name directly from CONFIG
                group = None
                if (
                    CONFIG["group_analysis"]["enabled"]
                    and CONFIG["group_analysis"]["group_column"]
                    in participant_data.columns
                    and not participant_data.empty
                ):
                    group = participant_data[
                        CONFIG["group_analysis"]["group_column"]
                    ].iloc[0]

                fit_results = fit_participant(
                    data_dicts,
                    simulation_code,
                    {k: v for k, v in var_desc.items() if v.get("learnable", False)},
                    target_variable,  # Pass the target variable
                    prediction_type,  # Pass the prediction type from config
                )

                if fit_results is not None:
                    # Keep participant_id in its original type
                    fit_results["participant_id"] = participant_id
                    fit_results["group"] = group  # Add the group information
                    fit_results["n_trials"] = len(data_dicts)  # Add number of trials
                    all_results.append(fit_results)
                    # Log successful fit results immediately
                    print(f"Successfully fit participant {participant_id}. Results: {fit_results}")

            except Exception as e:
                # Instead of just logging the error, add this participant to the skipped list
                skipped_participants.append(
                    {"participant_id": participant_id, "error": str(e)}
                )
                # Log the error but continue with other participants
                state.metadata.setdefault("fitting_warnings", []).append(
                    f"Error fitting participant {participant_id}: {str(e)}"
                )

        # Store information about skipped participants
        state.metadata["skipped_participants"] = skipped_participants
        state.metadata["num_skipped_participants"] = len(skipped_participants)

        # Log skipped participants information
        if skipped_participants:
            print(f"Skipped {len(skipped_participants)} participants: {skipped_participants}")

        # Check if we got any results after trying all participants
        if not all_results:
            state.metadata["fitting_error"] = "No successful parameter fits obtained"
            state.output.completion = f"Error: No successful parameter fits obtained. All {len(skipped_participants)} participants were skipped."
            return state

        # Calculate and store overall accuracy if this is a utility model
        if prediction_type.lower() == "utility":
            # Get all accuracy values from results
            accuracy_values = [
                result.get("accuracy", 0)
                for result in all_results
                if "accuracy" in result
            ]
            if accuracy_values:
                overall_accuracy = sum(accuracy_values) / len(accuracy_values)
                state.metadata["overall_accuracy"] = overall_accuracy

            # Calculate group-specific accuracy if group analysis is enabled
            if CONFIG["group_analysis"]["enabled"]:
                group_accuracies = {}
                for result in all_results:
                    # Use the 'group' field directly from the result dictionary
                    group = result.get("group")
                    accuracy = result.get("accuracy")
                    if group is not None and accuracy is not None:
                        group_accuracies.setdefault(group, []).append(accuracy)

                avg_group_accuracies = {}
                for group, acc_list in group_accuracies.items():
                    if acc_list:
                        avg_group_accuracies[group] = statistics.mean(acc_list)

                if avg_group_accuracies:
                    state.metadata["group_accuracies"] = avg_group_accuracies

        # --- Start: Calculate Group Parameter Averages ---
        if CONFIG["group_analysis"]["enabled"] and all_results:
            learnable_param_names = [
                name for name, info in var_desc.items() if info.get("learnable", False)
            ]
            group_param_values = {}  # Structure: {group: {param: [values]}}

            for result in all_results:
                group = result.get("group")
                if group is not None:
                    group_param_values.setdefault(group, {})
                    for param_name in learnable_param_names:
                        value = result.get(param_name)
                        # Ensure value is a number before adding
                        if isinstance(value, (int, float)):
                            # Check for NaN specifically for floats
                            if isinstance(value, float) and math.isnan(value):
                                continue  # Skip NaN values
                            group_param_values[group].setdefault(param_name, []).append(
                                value
                            )

            group_param_averages = {}  # Structure: {group: {param: average}}
            for group, params_dict in group_param_values.items():
                group_param_averages[group] = {}
                for param_name, values_list in params_dict.items():
                    if values_list:  # Check if list is not empty
                        try:
                            group_param_averages[group][param_name] = statistics.mean(
                                values_list
                            )
                        except statistics.StatisticsError:
                            # Handle case where list might be empty after filtering NaNs, though unlikely
                            group_param_averages[group][param_name] = None
                    else:
                        group_param_averages[group][param_name] = (
                            None  # No valid values found for this group/param
                        )

            # Store the calculated averages in metadata
            state.metadata["group_parameter_averages"] = group_param_averages
        # --- End: Calculate Group Parameter Averages ---

        # Store results and return success
        state.metadata["fitting_results"] = all_results

        # Include information about skipped participants in the completion message
        completion_message = ""
        if skipped_participants:
            completion_message = (
                f"Successfully fit parameters for {len(all_results)} participants. "
                f"Skipped {len(skipped_participants)} participants due to errors."
            )
        else:
            completion_message = (
                f"Successfully fit parameters for {len(all_results)} participants."
            )

        # Optionally add confirmation about group averages calculation
        if "group_parameter_averages" in state.metadata:
            completion_message += " Group parameter averages calculated."

        state.output.completion = completion_message

        return state

    return solve


@solver
def parameter_recovery_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        def generate_synthetic_data(
            simulation_code: str, true_params: Dict, participant_trials: List[Dict]
        ) -> List[Dict]:
            """Generate synthetic data using true parameters."""
            try:
                # Create local namespace for simulation
                local_vars = {"math": math, "random": random}
                exec(simulation_code, local_vars)

                if "simulate_model" not in local_vars:
                    raise ValueError("simulate_model function not found")

                # Generate utilities for all trials at once
                model_predictions = local_vars["simulate_model"](
                    participant_trials, **true_params
                )

                # Get prediction type and target variable
                prediction_type = CONFIG["comp_model_specifications"]["type"]
                target_variable = state.metadata.get("target_variable")

                # Generate synthetic decisions for each trial
                synthetic_data = []
                for trial, model_prediction in zip(
                    participant_trials, model_predictions
                ):
                    trial_copy = trial.copy()

                    # Handle different prediction types
                    if prediction_type.lower() == "utility":
                        # For utility models, convert to binary decision
                        p_accept = stable_logistic(model_prediction)
                        decision = random.random() < p_accept
                        trial_copy[target_variable] = 1 if decision else 0
                    else:
                        # For numerical prediction models, add Gaussian noise
                        noise_sd = CONFIG["parameter_recovery"].get("noise_sd", 0)
                        trial_copy[target_variable] = model_prediction + random.gauss(0, noise_sd)

                    synthetic_data.append(trial_copy)

                return synthetic_data

            except Exception as e:
                raise ValueError(f"Error generating synthetic data: {str(e)}")

        try:
            # Get necessary data from state
            fitting_results = state.metadata.get("fitting_results", [])
            simulation_code = state.metadata.get("simulation_code")
            var_desc = state.metadata.get("variable_descriptions", {})
            target_variable = state.metadata.get("target_variable")

            if not fitting_results:
                raise ValueError("No fitting results found")

            # Get learnable parameters
            learnable_params = {
                name: info
                for name, info in var_desc.items()
                if info.get("learnable", False)
            }

            param_ranges = {}
            for param in learnable_params:
                values = [
                    result[param] for result in fitting_results if param in result
                ]
                param_ranges[param] = {"min": min(values), "max": max(values)}

            # Read the full dataset once
            df = pd.read_csv(CONFIG["paths"]["dataset"])
            participant_ids = df["ID"].unique()

            n_iterations = CONFIG["parameter_recovery"]["n_iterations"]
            recovery_results = []

            # Get prediction type from config
            prediction_type = CONFIG["comp_model_specifications"]["type"]

            for i in range(n_iterations):
                # Randomly select a participant
                random_participant = random.choice(participant_ids)
                participant_trials = df[df["ID"] == random_participant].to_dict(
                    "records"
                )

                # Generate true parameters
                true_params = {
                    param: random.uniform(ranges["min"], ranges["max"])
                    for param, ranges in param_ranges.items()
                }

                # Generate synthetic data for all trials
                synthetic_data = generate_synthetic_data(
                    simulation_code, true_params, participant_trials
                )

                try:
                    # Fit parameters directly using synthetic data
                    recovered_params = fit_participant(
                        synthetic_data,
                        simulation_code,
                        learnable_params,
                        target_variable,  # Pass the target variable
                        prediction_type,  # Pass the prediction type
                    )

                    result = {
                        "iteration": i,
                        "participant_id": int(
                            random_participant
                        ),  # Convert to int to avoid [null]
                        **{f"true_{k}": v for k, v in true_params.items()},
                        **{f"recovered_{k}": recovered_params[k] for k in true_params},
                    }
                    recovery_results.append(result)

                except Exception as e:
                    state.metadata.setdefault("recovery_warnings", []).append(
                        f"Error in iteration {i}: {str(e)}"
                    )
                    continue

            # Create plots folder based on the output path from config
            plots_dir = Path(CONFIG["paths"]["plot_output"])
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Calculate correlations and create plots
            param_correlations = {}
            for param in learnable_params:
                # Handle lambda parameter name
                param_name = "lambda_param" if param == "lambda" else param

                true_vals = [r[f"true_{param}"] for r in recovery_results]
                recovered_vals = [r[f"recovered_{param}"] for r in recovery_results]

                corr, p_value = pearsonr(true_vals, recovered_vals)
                param_correlations[param] = {"r": corr, "p": p_value}

                plt.figure(
                    figsize=tuple(CONFIG["parameter_recovery"]["plots"]["figsize"])
                )
                plt.scatter(
                    true_vals,
                    recovered_vals,
                    alpha=CONFIG["parameter_recovery"]["plots"]["alpha"],
                )
                plt.xlabel(f"True {param_name}")
                plt.ylabel(f"Recovered {param_name}")
                plt.title(f"Parameter Recovery for {param_name} (r={corr:.3f})")

                plot_path = plots_dir / f"parameter_recovery_{param_name}.png"
                plt.savefig(str(plot_path))  # Convert Path to string for savefig
                plt.close()

            # Store correlations in metadata
            state.metadata["parameter_recovery"] = {
                "correlations": param_correlations,
                "plots_directory": str(plots_dir),
            }

            state.output.completion = f"Parameter recovery completed with {len(recovery_results)} successful iterations. Plots saved in {plots_dir}."

        except Exception as e:
            state.metadata["recovery_error"] = str(e)
            state.output.completion = f"Error in parameter recovery: {str(e)}"

        return state

    return solve


@solver
def model_validation_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            logging.info("Starting model validation...")

            # Get necessary data from state
            fitting_results = state.metadata.get("fitting_results", [])
            simulation_code = state.metadata.get("simulation_code")
            target_variable = state.metadata.get("target_variable")
            prediction_type = CONFIG["comp_model_specifications"]["type"]
            var_desc = state.metadata.get("variable_descriptions", {})

            if not all([fitting_results, simulation_code, target_variable]):
                logging.warning("Missing necessary data for model validation. Skipping.")
                state.output.completion = "Skipped model validation: Missing fitting results, simulation code, or target variable."
                return state

            # Read the full dataset once
            df = pd.read_csv(CONFIG["paths"]["dataset"])

            # Create plots folder
            plots_dir = Path(CONFIG["paths"]["plot_output"]) / "model_validation"
            plots_dir.mkdir(parents=True, exist_ok=True)

            def _mean_ignore_nan(values):
                """Helper to compute mean ignoring NaN values."""
                if not values:
                    return float('nan')
                clean_values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if not clean_values:
                    return float('nan')
                return statistics.mean(clean_values)

            # --- Helper function for simulation ---
            def run_simulation(participant_trials, learned_params):
                try:
                    local_vars = {"math": math, "random": random}
                    exec(simulation_code, local_vars)
                    simulate_model_func = local_vars.get("simulate_model")
                    if not simulate_model_func:
                        raise ValueError("simulate_model function not found in simulation_code")

                    model_predictions = simulate_model_func(participant_trials, **learned_params)

                    if prediction_type.lower() == "utility":
                        # For utility models, convert to binary decision
                        simulated_outcomes = []
                        for pred in model_predictions:
                            p_accept = stable_logistic(pred)
                            decision = 1 if random.random() < p_accept else 0
                            simulated_outcomes.append(decision)
                        return simulated_outcomes
                    else: # 'numerical_variable_estimation'
                        # For numerical models, the prediction is the outcome
                        return model_predictions

                except Exception as e:
                    logging.error(f"Error during simulation run: {e}")
                    return None

            # --- Perform validation for each participant ---
            participant_results = []
            for fit_res in fitting_results:
                participant_id = fit_res.get("participant_id")
                if participant_id is None:
                    continue
                
                participant_df = df[df["ID"] == participant_id]
                if participant_df.empty:
                    continue

                participant_trials = participant_df.to_dict("records")
                true_outcomes = participant_df[target_variable].tolist()

                learnable_params = {k: v for k, v in var_desc.items() if v.get("learnable")}
                learned_params = {p: fit_res[p] for p in learnable_params if p in fit_res}

                if len(learned_params) != len(learnable_params):
                    logging.warning(f"Skipping participant {participant_id}: Mismatch in learned parameters.")
                    continue

                simulated_outcomes = run_simulation(participant_trials, learned_params)

                if simulated_outcomes and len(true_outcomes) == len(simulated_outcomes):
                    group = fit_res.get("group")
                    participant_results.append({
                        "participant_id": participant_id,
                        "group": group,
                        "true_outcomes": true_outcomes,
                        "simulated_outcomes": simulated_outcomes,
                        "true_mean": _mean_ignore_nan(true_outcomes),
                        "simulated_mean": _mean_ignore_nan(simulated_outcomes)
                    })

            if not participant_results:
                 state.output.completion = "Model validation failed: No results generated for any participant."
                 state.metadata["validation_error"] = "No participant results for validation."
                 return state

            # --- Per-participant level plotting ---
            all_true_means = [r["true_mean"] for r in participant_results]
            all_simulated_means = [r["simulated_mean"] for r in participant_results]

            valid_pairs = [
                (t, s) for t, s in zip(all_true_means, all_simulated_means)
                if t is not None and s is not None and not (math.isnan(t) or math.isnan(s))
            ]

            participant_plot_path = None
            corr, p_val = float('nan'), float('nan')

            if valid_pairs:
                true_means, simulated_means = zip(*valid_pairs)
            
                plt.figure(figsize=(8, 8))
                plt.scatter(true_means, simulated_means, alpha=0.7)
                # Add y=x line
                min_val = min(min(true_means), min(simulated_means)) if true_means and simulated_means else 0
                max_val = max(max(true_means), max(simulated_means)) if true_means and simulated_means else 1
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
                
                plot_title_prefix = "Proportion of " if prediction_type.lower() == 'utility' else "Mean of "
                plt.xlabel(f"True {plot_title_prefix}{target_variable}")
                plt.ylabel(f"Simulated {plot_title_prefix}{target_variable}")
                plt.title("Per-Participant Model Validation")
                plt.legend()
                
                if len(true_means) > 1:
                    corr, p_val = pearsonr(true_means, simulated_means)
                
                plt.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', transform=plt.gca().transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                participant_plot_path = plots_dir / "participant_validation_scatter.png"
                plt.savefig(str(participant_plot_path))
                plt.close()
            else:
                logging.warning("No valid data pairs to plot for participant-level validation.")


            # --- Per-group level plotting ---
            group_plot_paths = {}
            if CONFIG["group_analysis"]["enabled"]:
                group_column = CONFIG["group_analysis"]["group_column"]
                groups = df[group_column].unique()

                for group in groups:
                    if pd.isna(group):
                        continue
                    
                    group_results = [r for r in participant_results if r["group"] == group]
                    if not group_results:
                        continue

                    true_group_means = [r["true_mean"] for r in group_results if r.get("true_mean") is not None and not math.isnan(r["true_mean"])]
                    simulated_group_means = [r["simulated_mean"] for r in group_results if r.get("simulated_mean") is not None and not math.isnan(r["simulated_mean"])]
                    
                    if not true_group_means and not simulated_group_means:
                        logging.warning(f"No valid data to plot for group '{group}'.")
                        continue

                    plt.figure(figsize=(10, 6))
                    if true_group_means:
                        plt.hist(true_group_means, bins=10, alpha=0.7, label='True Means', color='blue', density=True)
                    if simulated_group_means:
                        plt.hist(simulated_group_means, bins=10, alpha=0.7, label='Simulated Means', color='orange', density=True)
                    
                    plt.xlabel(f"Participant Mean {target_variable}")
                    plt.ylabel("Density")
                    plt.title(f"Group-level Validation for '{group}'")
                    plt.legend()
                    
                    group_plot_path = plots_dir / f"group_validation_hist_{group}.png"
                    plt.savefig(str(group_plot_path))
                    plt.close()
                    group_plot_paths[group] = str(group_plot_path)

            # --- Store results in metadata ---
            state.metadata["model_validation"] = {
                "participant_plot": str(participant_plot_path) if participant_plot_path else None,
                "group_plots": group_plot_paths,
                "correlation": {"r": corr if not math.isnan(corr) else None, "p": p_val if not math.isnan(p_val) else None}
            }
            
            completion_message = f"Model validation completed. Participant-level correlation r={corr:.3f}. Plots saved in {plots_dir}."
            state.output.completion = completion_message
            logging.info(completion_message)

        except Exception as e:
            logging.error(f"Error during model validation: {str(e)}", exc_info=True)
            state.metadata["validation_error"] = str(e)
            state.output.completion = f"Error in model validation: {str(e)}"
        
        return state

    return solve


def calculate_bic(log_likelihood: float, n_trials: int, k_params: int) -> float:
    """
    Calculate Bayesian Information Criterion (BIC) using log likelihood.

    Args:
        log_likelihood: Log likelihood of the model
        n_trials: Number of observations/trials
        k_params: Number of free parameters in the model

    Returns:
        float: BIC value
    """
    return -2 * log_likelihood + k_params * math.log(n_trials)


def calculate_bic_from_mse(mse: float, n_trials: int, k_params: int) -> float:
    """
    Calculate Bayesian Information Criterion (BIC) using MSE.

    For numerical prediction models, we convert MSE to log-likelihood assuming
    Gaussian errors with variance = MSE.

    Args:
        mse: Mean squared error of the model
        n_trials: Number of observations/trials
        k_params: Number of free parameters in the model

    Returns:
        float: BIC value
    """
    if mse <= 0:
        # Use a small positive value instead of returning infinity
        mse = 1e-10  # or some other small value appropriate for your scale
    # Convert MSE to log-likelihood under Gaussian error assumption
    # log(L) = -n/2 * log(2π) - n/2 * log(MSE) - n/2
    log_likelihood = (
        -n_trials / 2 * math.log(2 * math.pi)
        - n_trials / 2 * math.log(mse)
        - n_trials / 2
    )

    
    # Calculate BIC using the standard formula
    return -2 * log_likelihood + k_params * math.log(n_trials)


@metric
def bic() -> Metric:
    """Compute average BIC across all scores."""

    def metric(scores: list[Score]) -> float:
        bic_values = []
        for score in scores:
            if score.metadata and "bic_results" in score.metadata:
                bic_values.append(score.metadata["bic_results"]["average_bic"])
        return statistics.mean(bic_values) if bic_values else float("inf")

    return metric


def get_bic_summary(
    fitting_results: list, df: pd.DataFrame, k_params: int, group_analysis_enabled: bool
):
    """
    Given the list of fitting results and a DataFrame for the dataset,
    calculate an overall average BIC and (if enabled) the average BIC per group.

    Args:
        fitting_results: List of fitting results per participant
        df: DataFrame containing the dataset with group information
        k_params: Number of learnable parameters
        group_analysis_enabled: Whether group analysis is enabled

    Returns:
        tuple: (overall_avg, group_avg, individual_bics, group_bics)
            - overall_avg: overall average BIC (float)
            - group_avg: dict mapping each group to its average BIC
            - individual_bics: list of individual BIC values
            - group_bics: dict mapping group names to a list of BICs
    """
    bic_values = []
    group_bics = {}  # e.g., { "cocaine": [bic1, bic2, ...], "control": [bic3, bic4, ...] }

    for result in fitting_results:
        n_trials = result.get("n_trials")
        participant_id = result.get("participant_id")
        prediction_type = result.get(
            "prediction_type", "utility"
        )  # Default to utility if not specified

        # Skip if missing required data
        if n_trials is None or participant_id is None:
            logging.warning(f"Skipping BIC for participant {participant_id}: Missing n_trials or participant_id")
            continue

        # Calculate BIC based on prediction type
        if prediction_type.lower() == "utility":
            log_likelihood = result.get("log_likelihood")
            if log_likelihood is None:
                logging.warning(f"Skipping BIC for participant {participant_id} (utility): Missing log_likelihood")
                continue
            bic_value = calculate_bic(log_likelihood, n_trials, k_params)
        else:  # numerical prediction
            mse = result.get("mse")
            if mse is None:
                logging.warning(f"Skipping BIC for participant {participant_id} (numerical): Missing mse")
                continue
            bic_value = calculate_bic_from_mse(mse, n_trials, k_params)

        bic_values.append(bic_value)

        if group_analysis_enabled:
            participant_data = df[df["ID"] == participant_id]
            if not participant_data.empty:
                group = participant_data["group"].iloc[0]
                group_bics.setdefault(group, []).append(bic_value)

    overall_avg = statistics.mean(bic_values) if bic_values else None
    group_avg = {}
    if group_analysis_enabled:
        for group, values in group_bics.items():
            group_avg[group] = statistics.mean(values)

    return overall_avg, group_avg, bic_values, group_bics


@solver
def bic_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        logging.info("Starting BIC calculation...")
        try:
            # Ensure we have fitting results and variable descriptions
            fitting_results = state.metadata.get("fitting_results", [])
            var_desc = state.metadata.get("variable_descriptions", {})
            if not fitting_results:
                logging.warning("No fitting results found in metadata. Skipping BIC calculation.")
                state.output.completion = "Skipped BIC calculation: No fitting results."
                return state
            logging.info(f"Found {len(fitting_results)} fitting results.")

            # Determine the number of learnable parameters
            k_params = len([v for v in var_desc.values() if v.get("learnable", False)])
            logging.info(f"Number of learnable parameters (k): {k_params}")
            if k_params == 0:
                # Not much point calculating BIC if no parameters were fit
                logging.warning("No learnable parameters found. Skipping BIC calculation.")
                state.output.completion = (
                    "Skipped BIC calculation: No learnable parameters found."
                )
                return state

            # Read dataset (only once) to allow group analysis
            logging.debug(f"Reading dataset from {CONFIG['paths']['dataset']} for potential group analysis.")
            df = pd.read_csv(CONFIG['paths']['dataset'])
            # Activate group analysis only if the dataset has a "group" column
            # and the config explicitly enables it
            group_analysis_enabled = CONFIG["group_analysis"]["enabled"]
            group_column = CONFIG["group_analysis"]["group_column"]
            if group_analysis_enabled and group_column not in df.columns:
                logging.warning(f"Group analysis enabled in config, but column '{group_column}' not found in dataset. Disabling group BIC analysis.")
                group_analysis_enabled = False


            logging.info(f"Calling get_bic_summary with k={k_params} and group_analysis_enabled={group_analysis_enabled}")
            # Use helper to compute overall and per-group BIC values
            overall_avg, group_avg, individual_bics, group_bics = get_bic_summary(
                fitting_results, df, k_params, group_analysis_enabled # Pass modified enabled flag
            )

            if overall_avg is None:
                logging.warning("get_bic_summary returned None for overall_avg. No valid BIC values computed.")
                state.output.completion = "No valid BIC values computed."
                return state

            logging.info(f"BIC calculation successful. Average BIC: {overall_avg:.2f}")

            # Save the overall average BIC directly into the metadata
            state.metadata["average_bic"] = overall_avg

            # Get overall accuracy if it was calculated
            overall_accuracy = state.metadata.get("overall_accuracy")
            accuracy_message = ""
            if overall_accuracy is not None:
                accuracy_message = f"\nOverall Accuracy: {overall_accuracy:.4f}"
                logging.info(f"Overall accuracy found: {overall_accuracy:.4f}")


            # If group analysis is enabled, save each group's average BIC in the metadata
            if group_analysis_enabled:
                logging.info(f"Group BICs calculated: {group_avg}")
                for group, avg_value in group_avg.items():
                    # For example, if group is "cocaine", the key becomes "bic_cocaine"
                    state.metadata[f"bic_{group}"] = avg_value

            # Get prediction type from the first result (assuming all are the same type)
            prediction_type = (
                fitting_results[0].get("prediction_type", "utility")
                if fitting_results
                else "utility"
            )

            # Store full BIC results in metadata for detailed view
            bic_results_dict = {
                "average_bic": overall_avg,
                "individual_bics": individual_bics,
                "group_enabled": group_analysis_enabled, # Use the potentially modified flag
                "group_bics": group_bics if group_analysis_enabled else None,
                "num_parameters": k_params,
                "prediction_type": prediction_type,
                "bic_formula": "BIC = -2 * log_likelihood + k * log(n_trials)"
                if prediction_type == "utility"
                else "BIC calculated from MSE using Gaussian error assumption",
            }
            state.metadata["bic_results"] = bic_results_dict
            logging.debug(f"Stored bic_results in metadata: {bic_results_dict}")

            # Prepare a completion message showing the results
            messages = [f"Average BIC: {overall_avg:.2f}{accuracy_message}"]
            if group_analysis_enabled:
                for group, avg in group_avg.items():
                    messages.append(f"{group.title()} group BIC: {avg:.2f}")
            state.output.completion = "\n".join(messages)
            logging.info(f"BIC solver completion message: {state.output.completion}")


        except Exception as e:
            logging.error(f"Error during BIC calculation: {str(e)}", exc_info=True) # Log stack trace
            state.metadata["bic_error"] = str(e)
            state.output.completion = f"Error in BIC calculation: {str(e)}"
        return state

    return solve


@solver
def final_model_summary_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            # Get current model from metadata
            current_model = state.metadata.get("current_model", "")

            # If current_model is not available, reconstruct it from model_specification and model_summary
            if not current_model:
                model_spec = state.metadata.get("model_specification", "")
                model_summary = state.metadata.get("model_summary", "")
                target_variable = state.metadata.get("target_variable", "")
                prediction_type = state.metadata.get(
                    "prediction_type", CONFIG["comp_model_specifications"]["type"]
                )
                current_model = f"""Specification: {model_spec}
Summary: {model_summary}
Target Variable: {target_variable}
Prediction Type: {prediction_type}"""

            # Get BIC value if available
            bic_value = state.metadata.get("average_bic")
            if bic_value is not None:
                current_model += f"\nBIC: {bic_value}"

            # Get overall accuracy if available and if this is a utility model
            prediction_type = state.metadata.get("prediction_type", "")
            overall_accuracy = state.metadata.get("overall_accuracy")
            if prediction_type.lower() == "utility" and overall_accuracy is not None:
                current_model += f"\nOverall Accuracy: {overall_accuracy:.4f}"

            # Get and add group accuracy if available
            group_accuracies = state.metadata.get("group_accuracies")
            if group_accuracies:
                group_acc_str = "\nGroup Accuracies:"
                for group, acc in group_accuracies.items():
                    group_acc_str += f"\n- {group}: {acc:.4f}"
                current_model += group_acc_str

            # Get parameter recovery information if available
            param_recovery = state.metadata.get("parameter_recovery", {})
            correlations = param_recovery.get("correlations", {})

            # Add parameter recovery information to the current model if available
            if correlations:
                recovery_str = "\n\nParameter Recovery:"
                for param, values in correlations.items():
                    r_value = values.get("r", 0)
                    recovery_str += f"\n- {param}: r = {r_value:.3f}"

                # Add recovery info to the current model
                current_model += recovery_str

                # Also store recovery info in a more accessible format for multiple_runs.py
                state.metadata["recovery_summary"] = recovery_str.strip()

            # Get model validation information if available
            model_validation = state.metadata.get("model_validation", {})
            if model_validation:
                validation_corr = model_validation.get("correlation", {})
                r_value = validation_corr.get("r")
                if r_value is not None and not math.isnan(r_value):
                    validation_str = "\n\nModel Validation (True vs. Simulated Target):"
                    validation_str += (
                        f"\n- Per-participant correlation: r = {r_value:.3f}"
                    )
                    current_model += validation_str
                    state.metadata["validation_summary"] = validation_str.strip()

            # Update the previous_models list with the complete model information
            previous_models = state.metadata.get("previous_models", [])
            previous_models.append(current_model)
            state.metadata["previous_models"] = previous_models

            return state

        except Exception as e:
            state.metadata["final_summary_error"] = str(e)
            return state

    return solve


@scorer(metrics=[accuracy(), stderr(), bic()])
def verify() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Get the BIC results from the state metadata
        bic_results = state.metadata.get("bic_results", {})
        prediction_type = state.metadata.get(
            "prediction_type", CONFIG["comp_model_specifications"]["type"]
        )
        # Create explanation string
        explanation = f"Model Type: {prediction_type}\n\nBIC Results:\n"
        if bic_results:
            explanation += f"Average BIC: {bic_results.get('average_bic', 'N/A')}\n"
            explanation += (
                f"Number of parameters: {bic_results.get('num_parameters', 'N/A')}\n"
            )
            if "individual_bics" in bic_results:
                explanation += f"Individual BICs: {bic_results['individual_bics']}\n"
        else:
            explanation += "No BIC results found in metadata.\n"

        # Add group accuracy information if available
        group_accuracies = state.metadata.get("group_accuracies")
        if group_accuracies:
            explanation += "\nGroup Accuracies:\n"
            for group, acc in group_accuracies.items():
                explanation += f"- {group}: {acc:.4f}\n"

        return Score(
            value=CORRECT
            if bic_results
            else INCORRECT,  # Simple pass/fail based on presence of BIC results
            answer=str(bic_results.get("average_bic", "N/A")),
            explanation=explanation,
            metadata=state.metadata,  # Keep full metadata for our metrics
        )

    return score
