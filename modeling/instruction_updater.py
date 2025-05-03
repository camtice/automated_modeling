import logging
import traceback
from typing import Dict, Optional, Any
from inspect_ai.model import get_model, ChatMessage, GenerateConfig, ModelOutput
import re
import json


async def update_instructions(
    llm_name: str,
    previous_instructions: str,
    run_results: Dict[str, Any],
    complete_model_interaction: str,
    run_number: int,
    n_runs: int,
    previous_runs: list[Dict[str, Any]],
    num_previous_runs_to_include: int = None,
) -> str:
    """
    Generates updated instructions for the next modeling run using an LLM.

    Args:
        llm_name: Name of the language model to use (e.g., "openai/gpt-4o-mini").
        previous_instructions: The instructions used for the run that just completed.
        run_results: A dictionary containing key metrics from the completed run
                     (e.g., 'bic', 'accuracy', 'recovery_info').

    Returns:
        The newly generated instructions string. Returns previous_instructions if
        generation fails.
    """

    if not num_previous_runs_to_include:
        num_previous_runs_to_include = n_runs
    else:
        previous_runs = previous_runs[-num_previous_runs_to_include:]

    logging.info(f"Attempting to generate updated instructions using model: {llm_name}")
    try:
        # Prepare prompt inputs from results, handling potential None values
        bic_val_str = f"{run_results['bic']:.2f}"
        acc_val_str = f"{run_results['accuracy']:.3f}"
        rec_summary = run_results["recovery_info"]
        previous_runs_json = json.dumps(
            previous_runs, indent=2, default=str
        )  # default=str for safety

        # Construct the prompt for the LLM
        instruction_update_prompt = f"""You are an assistant helping refine instructions for a computational modeling LLM task. The LLM you are helping is stateless, i.e. it will not have access to it's previous interactions, so any information aside from the intial task and technical model descriptions will need to be provided in the your instructions.

        Based on the results of the previous run and the instructions used for that run, generate improved instructions for the *next* run. Aim to guide the main modeling LLM towards better performance (e.g., lower BIC, better parameter recovery, and higher accuracy). Please keep in mind that if any of the learnable parameters have a parameter recovery value of less than ~0.7, then the model is unusable. You will first be given the total context for the previous run, then reminded specifically of the instructions used for that run that you should update.

Here are a list of some recent previous models and their results for context:
<previous_runs>
{previous_runs_json}
</previous_runs>

Here is the best run so far in the previous models for context on what has worked so far (best run where the paramter recovery is above 0.7 for all learned parameters):
<best_run>
{return_best_run(previous_runs)}
</best_run>

Previous Complete Interaction:
<previous_interaction>
{complete_model_interaction}
</previous_interaction>

Previous Instructions:
<previous_instructions>
{previous_instructions}
</previous_instructions>

Previous Run Results:
<previous_run_results>
Average BIC: {bic_val_str}
Overall Accuracy (if applicable): {acc_val_str}
Parameter Recovery Summary:
{rec_summary}
</previous_run_results>

Generate *only* the text for the new instructions below for run {run_number + 1} of {n_runs}. Do any thinking between <think> and </think> tags. Remember to update the instructions to increase the accuracy, BIC, and paramater recovery. Be as specific or as vague as you would like. The model retrieving results are not able to use the previous interaction, so you should not include any information about the previous interaction in your response. You will also not be able to iterate on this response, so give your best advice each time. You should encourage out of the box thinking, since your counterpart likes to get stuck only with the most obvious models.
"""
        # Instantiate the LLM
        update_model = get_model(llm_name)

        # Make the LLM call
        output: ModelOutput = await update_model.generate(instruction_update_prompt)

        new_instructions = re.sub(
            r"<think>.*?</think>", "", output.completion.strip(), flags=re.DOTALL
        )

        if not new_instructions:  # Handle empty response
            logging.error("Instruction generation resulted in empty response.")
            raise ValueError(
                "Empty response received from instruction generation model"
            )

        logging.info("Successfully generated updated instructions.")
        return new_instructions

    except Exception as e:
        logging.error(f"Failed to update instructions: {e}", exc_info=True)
        raise e


def return_best_run(previous_runs: list[Dict[str, Any]]):
    # return the run with the lowest bic and paramter recovery above 0.7 for all variables
    best_run = None
    best_bic = float("inf")

    for run in previous_runs:
        recovery_info_str = run.get("recovery_info", "")
        # Find all floating point numbers following 'r = '
        recovery_values = re.findall(r"r\s*=\s*(\d+\.\d+)", recovery_info_str)
        # Convert found strings to floats
        recovery_values_float = [float(val) for val in recovery_values]

        # Check if all recovery values are above 0.7 (and if any values were found)
        all_recovery_above_threshold = bool(recovery_values_float) and all(
            val > 0.7 for val in recovery_values_float
        )

        if all_recovery_above_threshold:
            run_bic = run.get("bic")
            # Ensure BIC exists and is comparable
            if run_bic is not None and isinstance(run_bic, (int, float)):
                if run_bic < best_bic:
                    best_bic = run_bic
                    best_run = run

    if best_run is None:
        best_run = []

    return best_run
