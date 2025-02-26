from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import (
    match,
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
from inspect_ai.model import ModelOutput
from inspect_ai.util import ExecResult, sandbox
import pandas as pd
from typing import Dict, Any, List
import json
import random
import math
import traceback
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from model_utils import fit_participant, stable_logistic, get_dataset_info
import statistics
import yaml
from pathlib import Path
import shutil
from inspect_ai.model import GenerateConfig


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and required fields."""
    required_fields = {
        "paths": ["dataset", "plot_output"],
        "model": ["name"],
        "parameter_recovery": ["n_iterations"],
        "system": ["prompt"],
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

        for field in fields:
            if field not in config[section]:
                raise ValueError(
                    f"Missing required field '{field}' in section '{section}'"
                )

    # Validate paths exist
    dataset_path = Path(config["paths"]["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")


def load_config(config_path: str = None) -> dict:
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

        # Format previous models with tags before generating
        previous_models = ""
        for i, model in enumerate(state.metadata.get("previous_models", []), 1):
            previous_models += (
                f"\nModel {i}:\n<previous_model_{i}>\n{model}\n</previous_model_{i}>\n"
            )

        # Combine the task description with desired output description
        prompt = f"""
Task Description: {state.metadata["task_description"]}

Desired Output Specification: {state.metadata["output_description"]}

Previous Models:{previous_models}

Please think through this step by step, then provide your model specification and variable descriptions.
""".strip()
        # Update the user prompt in state
        state.user_prompt.text = prompt

        # Generate using custom model
        output = await custom_model.generate(state.messages)
        state.output = output
        state.messages.append(output.message)
        # Log the complete interaction for debugging
        state.metadata.setdefault("complete_model_interaction", []).append(
            {
                "solver": "model_design_solver",
                "input": state.user_prompt.text,
                "output": output.completion,
            }
        )

        # Extract model, variable descriptions, and summary using regex
        import re

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
        state.metadata["model_summary"] = summary  # Add summary to metadata
        state.metadata["full_reasoning"] = state.output.completion

        # Store the raw model (without tags) in previous_models list
        current_model = f"""Specification: {model}
Summary: {summary}"""

        previous_models = state.metadata.get("previous_models", [])
        previous_models.append(current_model)
        state.metadata["previous_models"] = previous_models

        return state

    return solve


@solver
def simulate_model_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Extract model, variables, and dataset info from previous state
                model_spec = state.metadata.get("model_specification")
                var_desc = state.metadata.get("variable_descriptions", {})
                dataset_info = state.metadata.get("dataset_info")
                learnable_params = state.metadata.get("learnable_parameters", {})

                # Format variable descriptions for the prompt
                var_desc_formatted = json.dumps({"variables": var_desc}, indent=2)

                # Update the prompt to use the formatted variable descriptions
                prompt = f"""
                Write Python code to simulate the following mathematical model using only Python standard library (no pandas or numpy).
                The code should take a list of dictionaries as input (each dictionary representing a trial) and return the utility values for each trial.

                IMPORTANT: Use these exact parameter names for learnable parameters: {list(learnable_params.keys())}

                Please write a function called `simulate_model` that:
                1. Takes a list of dictionaries as input (each dictionary contains trial data)
                2. Has parameter names that matches the variable descriptions exactly
                3. Implements the mathematical model above using only standard Python math operations
                4. Returns a list of utility values for each trial

                Your code will be implemented within the following code block:

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
    utility_values = []
    for trial in trial_data:
        E = trial.get("environmental_cue", 0)
        u1, u2 = random.random(), random.random()
        N = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        utility_values.append(beta + (epsilon * E) + (eta * N))
    return utility_values
</YOUR RESPONSE>

Now, create a simulate_model function for the following model incorporating the information provided. DO NOT OUTPUT ANYTHING AFTER YOU RETURN THE UTLITY VALUES.
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
                    }
                )

                # Clean up the generated code
                simulation_code = state.output.completion

                # Extract the complete function definition including return statement
                import re

                # Regex to capture the entire function including def and return
                function_match = re.search(
                    r"(def\s+simulate_model\s*\(.*?\):.*?return\s+utility_values)",
                    simulation_code,
                    re.DOTALL,
                )
                if function_match:
                    # Assign the entire function to simulation_code
                    simulation_code = function_match.group(1).strip()
                else:
                    raise ValueError(
                        "Could not find complete simulate_model function in generated code"
                    )
                # Eventually from here to the end of the function can be deletd, kept for now for debugging
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

                    # Modify the sandbox execution code block to ensure clean JSON output
                    code = f"""
import json
import math

{simulation_code}

# Data as list of dictionaries
data = json.loads('''{json.dumps(data_dicts_clean)}''')

try:
    # Get results for the data
    results = simulate_model(data)
    # Ensure single-line JSON output
    print(json.dumps({{"results": results}}).strip())
except Exception as e:
    print(json.dumps({{"error": str(e)}}).strip())
"""

                    # Execute the code
                    result = await sandbox().exec(
                        cmd=["python", "-c", code],
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
                                state.output.completion = (
                                    f"Error in simulation: {output_data['error']}"
                                )
                            else:
                                simulation_results = output_data["results"]
                                # Flatten the results into a single list
                                # state.metadata["simulation_results"] = simulation_results
                                state.metadata["simulation_code"] = simulation_code
                                state.output.completion = str(simulation_results)
                        except json.JSONDecodeError as e:
                            state.metadata["simulation_error"] = (
                                f"JSON parsing error: {str(e)}\nOutput: {result.stdout}"
                            )
                            state.output.completion = (
                                f"Error parsing simulation output: {str(e)}"
                            )
                    else:
                        state.metadata["simulation_error"] = result.stderr
                        state.metadata["simulation_code"] = simulation_code
                        state.output.completion = (
                            f"Error in simulation: {result.stderr}"
                        )

                except Exception as e:
                    error_msg = str(e)
                    state.metadata["simulation_error"] = error_msg
                    state.output.completion = f"Error in simulation: {error_msg}"

                return state

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    state.metadata["simulation_error"] = (
                        f"Failed after {max_retries} attempts: {str(e)}"
                    )
                    state.output.completion = f"Error in simulation: {str(e)}"
                    return state

                # Try regenerating on failure and log the interaction
                current_prompt = state.user_prompt.text
                state = await generate(state)
                state.metadata.setdefault("complete_model_interaction", []).append(
                    {
                        "solver": "simulate_model_solver",
                        "input": current_prompt,
                        "output": state.output.completion,
                    }
                )
                continue

        return state

    return solve


@solver
def parameter_fitting_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        max_retries = CONFIG["fitting"]["max_retries"]
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Get necessary data from state
                fitting_results = state.metadata.get("fitting_results", [])
                simulation_code = state.metadata.get("simulation_code")
                var_desc = state.metadata.get("variable_descriptions", {})

                # Update: Use config for dataset path
                df = pd.read_csv(CONFIG["paths"]["dataset"])

                all_results = []
                participant_ids = df["ID"].unique()

                # Fit for each participant
                for participant_id in participant_ids:
                    try:
                        participant_data = df[df["ID"] == participant_id]
                        data_dicts = participant_data.to_dict("records")

                        fit_results = fit_participant(
                            data_dicts,
                            simulation_code,
                            {
                                k: v
                                for k, v in var_desc.items()
                                if v.get("learnable", False)
                            },
                        )

                        if fit_results is not None:
                            fit_results["participant_id"] = participant_id
                            fit_results["n_trials"] = len(
                                data_dicts
                            )  # Add number of trials
                            all_results.append(fit_results)

                    except Exception as e:
                        state.metadata.setdefault("fitting_warnings", []).append(
                            f"Error fitting participant {participant_id}: {str(e)}"
                        )
                        continue

                # Check if we got any results
                if not all_results:
                    raise ValueError("No successful parameter fits obtained")

                state.metadata["fitting_results"] = all_results
                state.output.completion = (
                    f"Successfully fit parameters for {len(all_results)} participants"
                )
                return state

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    state.metadata["fitting_error"] = (
                        f"Failed after {max_retries} attempts: {str(e)}"
                    )
                    state.output.completion = f"Error in parameter fitting: {str(e)}"
                    return state

                # Try regenerating on failure and log the interaction
                current_prompt = state.user_prompt.text
                state = await generate(state)
                state.metadata.setdefault("complete_model_interaction", []).append(
                    {
                        "solver": "parameter_fitting_solver",
                        "input": current_prompt,
                        "output": state.output.completion,
                    }
                )
                continue

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
                utilities = local_vars["simulate_model"](
                    participant_trials, **true_params
                )

                # Generate synthetic decisions for each trial
                synthetic_data = []
                for trial, utility in zip(participant_trials, utilities):
                    trial_copy = trial.copy()
                    p_accept = stable_logistic(utility)
                    decision = random.random() < p_accept
                    trial_copy["accept"] = 1 if decision else 0
                    synthetic_data.append(trial_copy)

                return synthetic_data

            except Exception as e:
                raise ValueError(f"Error generating synthetic data: {str(e)}")

        try:
            # Get necessary data from state
            fitting_results = state.metadata.get("fitting_results", [])
            simulation_code = state.metadata.get("simulation_code")
            var_desc = state.metadata.get("variable_descriptions", {})

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
                        synthetic_data, simulation_code, learnable_params
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


def calculate_bic(log_likelihood: float, n_trials: int, k_params: int) -> float:
    """
    Calculate Bayesian Information Criterion (BIC).

    Args:
        log_likelihood: Log likelihood of the model
        n_trials: Number of observations/trials
        k_params: Number of free parameters in the model

    Returns:
        float: BIC value
    """
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
        log_likelihood = result.get("log_likelihood")
        n_trials = result.get("n_trials")
        participant_id = result.get("participant_id")

        if (log_likelihood is None) or (n_trials is None) or (participant_id is None):
            continue

        bic_value = calculate_bic(log_likelihood, n_trials, k_params)
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
        try:
            # Ensure we have fitting results and variable descriptions
            fitting_results = state.metadata.get("fitting_results", [])
            var_desc = state.metadata.get("variable_descriptions", {})
            if not fitting_results:
                return state

            # Determine the number of learnable parameters
            k_params = len([v for v in var_desc.values() if v.get("learnable", False)])
            if k_params == 0:
                # Not much point calculating BIC if no parameters were fit
                state.output.completion = (
                    "No learnable parameters found to calculate BIC."
                )
                return state

            # Read dataset (only once) to allow group analysis
            df = pd.read_csv(CONFIG["paths"]["dataset"])
            # Activate group analysis only if the dataset has a "group" column
            # and the config explicitly enables it
            group_enabled = CONFIG.get("group_analysis", {}).get("enabled", False) and (
                "group" in df.columns
            )

            # Use helper to compute overall and per-group BIC values
            overall_avg, group_avg, individual_bics, group_bics = get_bic_summary(
                fitting_results, df, k_params, group_enabled
            )

            if overall_avg is None:
                state.output.completion = "No valid BIC values computed."
                return state

            # Save the overall average BIC directly into the metadata
            state.metadata["average_bic"] = overall_avg

            # If group analysis is enabled, save each group's average BIC in the metadata
            if group_enabled:
                for group, avg_value in group_avg.items():
                    # For example, if group is "cocaine", the key becomes "bic_cocaine"
                    state.metadata[f"bic_{group}"] = avg_value

            # Store BIC values in state.output.metrics for CSV output
            state.output.metrics["average_bic"] = overall_avg
            if group_enabled:
                for group, value in group_avg.items():
                    state.output.metrics[f"bic_{group}"] = value

            # Store full BIC results in metadata for detailed view
            state.metadata["bic_results"] = {
                "average_bic": overall_avg,
                "individual_bics": individual_bics,
                "group_enabled": group_enabled,
                "group_bics": group_bics if group_enabled else None,
                "num_parameters": k_params,
                "bic_formula": "BIC = -2 * log_likelihood + k * log(n_trials)",
            }

            # Prepare a completion message showing the results
            messages = [f"Average BIC: {overall_avg:.2f}"]
            if group_enabled:
                for group, avg in group_avg.items():
                    messages.append(f"{group.title()} group BIC: {avg:.2f}")
            state.output.completion = "\n".join(messages)

        except Exception as e:
            state.metadata["bic_error"] = str(e)
            state.output.completion = f"Error in BIC calculation: {str(e)}"
        return state

    return solve


@scorer(metrics=[accuracy(), stderr(), bic()])
def verify() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Get the BIC results from the state metadata
        bic_results = state.metadata.get("bic_results", {})

        # Create explanation string
        explanation = "BIC Results:\n"
        if bic_results:
            explanation += f"Average BIC: {bic_results.get('average_bic', 'N/A')}\n"
            explanation += (
                f"Number of parameters: {bic_results.get('num_parameters', 'N/A')}\n"
            )
            if "individual_bics" in bic_results:
                explanation += f"Individual BICs: {bic_results['individual_bics']}\n"
        else:
            explanation += "No BIC results found in metadata.\n"

        return Score(
            value=CORRECT
            if bic_results
            else INCORRECT,  # Simple pass/fail based on presence of BIC results
            answer=str(bic_results.get("average_bic", "N/A")),
            explanation=explanation,
            metadata=state.metadata,  # Keep full metadata for our metrics
        )

    return score


@task
def design_model(
    task_description: str = TASK_DESCRIPTION,
    output_description: str = MODEL_OUTPUT_DESCRIPTION,
    dataset_path: str = CONFIG["paths"]["dataset"],
) -> Task:
    """
    Task to generate a computational model specification based on descriptions.

    Args:
        task_description: Detailed description of the task/problem to solve
        output_description: Description of desired model outputs and characteristics
        dataset_path: Path to CSV file containing the dataset structure
    """
    # Get dataset information
    dataset_info = ""
    info = get_dataset_info(dataset_path)
    if "error" not in info:
        dataset_info = "\nDataset Structure:\n"
        dataset_info += "Variables available:\n"
        for var, dtype in info["data_types"].items():
            dataset_info += f"- {var} ({dtype})\n"
        dataset_info += f"\nNumber of observations: {info['n_rows']}"

    # Format task description with dataset info
    formatted_task = task_description.format(dataset_info=dataset_info)

    return Task(
        sandbox="docker",
        dataset=[
            Sample(
                input="Generate model specification",
                target="",  # No specific target since this is a generative task
                metadata={
                    "task_description": formatted_task,
                    "output_description": output_description,
                    "dataset_info": dataset_info,
                },
            )
        ],
        solver=[
            system_message(CONFIG["system"]["prompt"]),  # Use system prompt from CONFIG
            model_design_solver(),
            simulate_model_solver(),
            parameter_fitting_solver(),
            parameter_recovery_solver(),
            bic_solver(),
        ],
        scorer=verify(),
    )
