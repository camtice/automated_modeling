# Currently in use.


# The current issue is that I use global variables in the other script which means that I will
# need to do a medium amount of work cleaning those up and ensuring they are modular.

# It also looks like I alternate between using global variables and pulling from the metadata, which
# just does not seem lke good practice. This is something that I would like to clean up.


import json
import re

from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.model import get_model, GenerateConfig


@solver
def model_design_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Now you can directly use CONFIG here
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
Your model should predict the utility of a binary choice. The utility will be converted to a probability using a logistic function, and then used to predict binary decisions.
"""
        elif prediction_type.lower() == "numerical_variable_estimation":
            prediction_type_info = """
Your model should directly predict a numerical value (not a binary choice). The model's predictions will be compared to actual values using mean squared error.
"""
        else:
            raise ValueError(f"Invalid prediction type: {prediction_type}")

        # Combine the task description with desired output description
        output_description = state.metadata["output_description"]

        # Check if output_description already contains "Previous Models:"
        if "Previous Models:" in output_description:
            # If it does, don't add previous_models again as they're already included
            prompt = f"""
Task Description: {state.metadata["task_description"]}

Desired Output Specification: {output_description}

Model Type: {prediction_type_info}

Please think through this step by step, then provide your model specification and variable descriptions.
""".strip()
        else:
            # If not, add previous_models as before
            prompt = f"""
Task Description: {state.metadata["task_description"]}

Desired Output Specification: {output_description}

Model Type: {prediction_type_info}

Previous Models:{previous_models if previous_models else "Not Provided"}

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
