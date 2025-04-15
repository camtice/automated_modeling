# Check the order of previous models to make sure they are by time, and then order and return best

from inspect_ai import eval, Task
from initial_generation import (
    MODEL_OUTPUT_DESCRIPTION,
    TASK_DESCRIPTION,
    CONFIG,
)
from initial_generation import (
    model_design_solver,
    simulate_model_solver,
    parameter_fitting_solver,
    parameter_recovery_solver,
    bic_solver,
    final_model_summary_solver,
    verify,
    load_config as load_initial_config,
    CONFIG as INITIAL_CONFIG,
)
from inspect_ai.dataset import Sample
from inspect_ai.solver import system_message
from model_utils import get_dataset_info
import pandas as pd
import os
import json
from datetime import datetime
import shutil
from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
import traceback
import asyncio
from inspect_ai.model import GenerateConfig

from instruction_updater import update_instructions


def design_model_local(
    task_description: str,
    output_description: str,
    instructions: str,
    dataset_path: str,
    system_prompt: str,
) -> Task:
    """
    Defines the Task for a single model generation run using local sandbox.

    Args:
        task_description: Detailed description of the task/problem to solve for this run.
        output_description: Description of desired model outputs for this run.
        instructions: Specific instructions for the LLM for this run.
        dataset_path: Path to the CSV dataset file.
        system_prompt: The system prompt to use for the main LLM.
    """
    # Get dataset information (consider making this more robust if path changes)
    dataset_info_text = ""
    try:
        info = get_dataset_info(dataset_path)
        if "error" not in info:
            dataset_info_text = "\nDataset Structure:\n"
            dataset_info_text += "Variables available:\n"
            for var, dtype in info["data_types"].items():
                dataset_info_text += f"- {var} ({dtype})\n"
            dataset_info_text += f"\nNumber of observations: {info['n_rows']}"
        else:
            logging.warning(f"Could not get dataset info: {info['error']}")
    except FileNotFoundError:
        logging.error(f"Dataset file not found at: {dataset_path}")
        # Decide how to handle - maybe raise error or return empty info?
        # For now, continue with empty info
        pass

    # Format task description with dataset info (if available)
    formatted_task = task_description.format(dataset_info=dataset_info_text)

    return Task(
        sandbox="local",
        dataset=[
            Sample(
                input="Generate model specification",
                target="",
                metadata={
                    "task_description": formatted_task,
                    "output_description": output_description,
                    "instructions": instructions,
                    "dataset_info": dataset_info_text,
                },
            )
        ],
        solver=[
            system_message(system_prompt),
            model_design_solver(),
            simulate_model_solver(),
            parameter_fitting_solver(),
            parameter_recovery_solver(),
            bic_solver(),
            final_model_summary_solver(),
        ],
        scorer=verify(),
    )


def load_run_config(config_path: Optional[str] = None) -> dict:
    """Load the configuration specific to the multiple runs script."""
    base_dir = Path(__file__).parent
    if not config_path:
        config_path = base_dir / "configs" / "multiple_runs_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Multiple runs config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Store the actual config path used for later copying
    config["_config_path"] = str(config_path)
    return config


def load_text_file(file_path: Path) -> str:
    """Loads content from a text file."""
    if not file_path.exists():
        logging.warning(f"Text file not found: {file_path}, returning empty string.")
        return ""
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
        return ""


def setup_logging(log_file: Path) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    # Suppress overly verbose loggers if needed
    for logger_name in ["urllib3", "docker", "matplotlib"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)


def setup_experiment_directory(config: dict) -> tuple[Path, str]:
    """Set up experiment directory structure and return paths."""
    exp_base_name = config["experiment"]["base_name"]
    root_dir = Path(config["storage"]["root_dir"])
    version = 0
    while True:
        exp_dir = root_dir / f"{exp_base_name}_v{version}"
        if not exp_dir.exists():
            break
        version += 1

    version_str = f"v{version}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create essential subdirectories
    data_dir = exp_dir / config["storage"]["structure"]["data"]
    raw_data_dir = data_dir / "raw"
    configs_dir = exp_dir / "configs"
    prompts_dir = configs_dir / "prompts"

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    log_file = exp_dir / "experiment.log"
    setup_logging(log_file)

    logging.info(f"Starting experiment {version_str} in directory: {exp_dir}")

    # Load environment variables
    base_dir = Path(__file__).parent
    env_path = base_dir.parent / ".env"
    load_dotenv(env_path)
    logging.info(f"Checked for .env file at: {env_path}")
    inspect_eval_model_env = os.getenv("INSPECT_EVAL_MODEL")
    if inspect_eval_model_env:
        logging.info(
            f"INSPECT_EVAL_MODEL environment variable: {inspect_eval_model_env}"
        )
    else:
        logging.warning("INSPECT_EVAL_MODEL environment variable not set.")

    # Save experiment metadata
    metadata = {
        "version": version_str,
        "experiment_config": config,
        "datetime": datetime.now().isoformat(),
        "environment": {"INSPECT_EVAL_MODEL": inspect_eval_model_env or "Not Set"},
    }
    with open(exp_dir / config["storage"]["file_formats"]["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy config files
    script_config_dir = base_dir / "configs"
    if "_config_path" in config:
        try:
            shutil.copy2(
                config["_config_path"], configs_dir / Path(config["_config_path"]).name
            )
        except Exception as e:
            logging.warning(f"Could not copy specific run config: {e}")

    default_config_path = script_config_dir / "default_config.yaml"
    if default_config_path.exists():
        shutil.copy2(default_config_path, configs_dir / "default_config.yaml")
    else:
        logging.warning("default_config.yaml not found, not copied.")

    # Copy prompt files
    prompt_files_to_copy = [
        "model_output_description.txt",
        "task_description.txt",
        "instructions.txt",
        "system_prompt.txt",
    ]
    script_prompts_dir = script_config_dir / "prompts"
    for prompt_file in prompt_files_to_copy:
        source_path = script_prompts_dir / prompt_file
        if source_path.exists():
            shutil.copy2(source_path, prompts_dir / prompt_file)
        else:
            logging.warning(f"Prompt file not found: {source_path}, not copied.")

    return exp_dir, version_str


def run_multiple_evaluations(config_path: Optional[str] = None):
    """Run multiple evaluations, potentially updating instructions between runs."""
    # Load configurations
    run_config = load_run_config(config_path)
    initial_config = load_initial_config()

    # --- Determine number of runs ---
    n_runs = run_config["experiment"].get("n_runs")
    logging.info(f"Number of runs: {n_runs}")

    # --- Setup Experiment ---
    exp_dir, version = setup_experiment_directory(run_config)
    print(f"Experiment {version} started in: {exp_dir}")
    print(f"Running {n_runs} evaluations...")

    # --- Load Initial Prompts ---
    script_prompts_dir = Path(__file__).parent / "configs" / "prompts"
    task_description_template = load_text_file(
        script_prompts_dir / "task_description.txt"
    )
    model_output_description_base = load_text_file(
        script_prompts_dir / "model_output_description.txt"
    )
    initial_instructions = load_text_file(script_prompts_dir / "instructions.txt")
    system_prompt = load_text_file(script_prompts_dir / "system_prompt.txt")

    if not all(
        [
            task_description_template,
            model_output_description_base,
            initial_instructions,
            system_prompt,
        ]
    ):
        logging.error(
            "One or more essential prompt files are missing or empty. Exiting."
        )
        return None

    # --- Get LLM Details for Instruction Updates ---
    llm_name = initial_config.get("model", {}).get("name")
    if not llm_name:
        logging.error(
            "LLM name not found in default_config.yaml under model.name. Cannot update instructions."
        )
    else:
        update_instructions_enabled = run_config["prompt_updater"][
            "update_instructions_dynamically"
        ]
        logging.info(f"Instruction updates enabled: {update_instructions_enabled}")
        if update_instructions_enabled:
            logging.info(f"Using LLM for updates: {llm_name}")

    # --- Initialize Loop Variables ---
    instructions_for_next_run = initial_instructions
    results_list = []
    previous_models_data = []

    # --- Dataset Path ---
    dataset_path = initial_config.get("paths", {}).get("dataset")
    if not dataset_path or not Path(dataset_path).exists():
        logging.error(
            f"Dataset path '{dataset_path}' from default_config.yaml is invalid or missing."
        )
        print(f"Error: Dataset path '{dataset_path}' is invalid or missing. See logs.")
        return None

    # --- Main Evaluation Loop ---
    progress_bar = tqdm(range(n_runs), desc="Running evaluations", unit="run")
    os.environ["INSPECT_QUIET"] = "1"

    for i in progress_bar:
        run_number = i + 1
        logging.info(f"===== Starting Evaluation {run_number}/{n_runs} =====")
        logging.info(f"Using Instructions:\n---\n{instructions_for_next_run}\n---")

        run_dir = exp_dir / "data" / "raw" / f"run_{run_number}"
        plots_dir = run_dir / run_config["storage"]["structure"]["plots"]
        run_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        current_run_results = {}

        try:
            # --- Prepare Output Description ---
            output_description_for_run = model_output_description_base
            if (
                run_config["experiment"].get("consider_previous_models", True)
                and previous_models_data
            ):
                previous_models_text = "Previous Models:\n"
                for idx, model_data in enumerate(previous_models_data, 1):
                    previous_models_text += f"Model {idx}:\n"
                    spec = model_data.get("specification", "N/A")
                    summary = model_data.get("summary", "N/A")
                    bic = model_data.get("bic")
                    rec_info = model_data.get("recovery_info", "")

                    previous_models_text += f"Specification: {spec}\n"
                    previous_models_text += f"Summary: {summary}\n"

                    if (
                        run_config["experiment"].get("include_bic_in_summary", False)
                        and bic is not None
                    ):
                        previous_models_text += f"BIC: {bic:.2f}\n"
                    if rec_info and run_config["experiment"].get(
                        "include_recovery_in_summary", True
                    ):
                        previous_models_text += f"{rec_info}\n"

                output_description_for_run = (
                    model_output_description_base.rstrip()
                    + "\n\n"
                    + run_config["experiment"].get("previous_models_prompt", "").strip()
                    + "\n\n"
                    + previous_models_text.rstrip()
                )

            # --- Run the Evaluation Synchronously ---
            task = design_model_local(
                task_description=task_description_template,
                output_description=output_description_for_run,
                instructions=instructions_for_next_run,
                dataset_path=dataset_path,
                system_prompt=system_prompt,
            )
            log_list = eval(task, trace=False, progress=False)
            log = log_list[0]

            # --- Process Results ---
            if log.samples and log.samples[0].metadata:
                md = log.samples[0].metadata

                bic_value = md.get("average_bic")
                bic_control = md.get("bic_Control")
                bic_cocaine = md.get("bic_Cocaine")
                model_spec = md.get("model_specification", "")
                model_summary = md.get("model_summary", "")
                recovery_correlations = md.get("parameter_recovery", {}).get(
                    "correlations", {}
                )
                recovery_summary_str = md.get("recovery_summary", "")
                overall_accuracy = md.get("overall_accuracy")
                complete_model_interaction = md.get("complete_model_interaction")

                current_run_results = {
                    "specification": model_spec,
                    "summary": model_summary,
                    "bic": bic_value,
                    "bic_control": bic_control,
                    "bic_cocaine": bic_cocaine,
                    "recovery_info": recovery_summary_str,
                    "accuracy": overall_accuracy,
                }
                previous_models_data.append(current_run_results)

                # Move plots
                plots_source_dir = Path(
                    initial_config.get("paths", {}).get(
                        "plot_output", "param_recovery_plots"
                    )
                )
                if plots_source_dir.exists():
                    try:
                        for file in os.listdir(plots_source_dir):
                            shutil.move(
                                str(plots_source_dir / file), str(plots_dir / file)
                            )
                        if not os.listdir(plots_source_dir):
                            os.rmdir(plots_source_dir)
                    except Exception as plot_err:
                        logging.warning(
                            f"Could not move/remove plot directory {plots_source_dir}: {plot_err}"
                        )
                else:
                    logging.info(
                        f"Plot source directory '{plots_source_dir}' not found, skipping move."
                    )

                # Create row for results table
                row_data = {
                    "run_number": run_number,
                    "average_bic": bic_value,
                    "bic_control": bic_control,
                    "bic_cocaine": bic_cocaine,
                    "overall_accuracy": overall_accuracy,
                    "model_specification": model_spec,
                    "model_summary": model_summary,
                    "version": version,
                    "instructions_used": instructions_for_next_run,
                }
                for param, data in recovery_correlations.items():
                    r_value = data.get("r")
                    row_data[f"{param}_recovery"] = r_value
                results_list.append(row_data)

                # Save individual run metadata
                with open(run_dir / "metadata.json", "w") as f:
                    json.dump(md, f, indent=2, default=str)

                # Update progress bar and print summary
                recovery_str = ", ".join(
                    [
                        f"{p}={v['r']:.2f}" if v.get("r") is not None else f"{p}=N/A"
                        for p, v in recovery_correlations.items()
                    ]
                )
                bic_str = f"BIC={bic_value:.2f}" if bic_value is not None else "BIC=N/A"
                if bic_control is not None:
                    bic_str += f", Ctrl={bic_control:.2f}"
                if bic_cocaine is not None:
                    bic_str += f", Coc={bic_cocaine:.2f}"
                acc_str = (
                    f", Acc={overall_accuracy:.2f}"
                    if overall_accuracy is not None
                    else ""
                )
                model_info = (
                    f"Run {run_number}: {bic_str}{acc_str}, Recov: {recovery_str}"
                )
                progress_bar.set_postfix_str(model_info, refresh=True)
                print(f"\nRun {run_number} complete - {model_info}")

            else:
                logging.warning(f"No sample metadata found for run {run_number}")
                results_list.append(
                    {
                        "run_number": run_number,
                        "average_bic": None,
                        "model_specification": "No metadata",
                        "version": version,
                    }
                )
                progress_bar.set_postfix_str(
                    f"Run {run_number}: No metadata", refresh=True
                )
                print(f"\nRun {run_number} complete - No metadata found")
                current_run_results = None

            # --- Update Instructions for Next Run ---
            if update_instructions_enabled and current_run_results and (i < n_runs - 1):
                try:
                    instructions_for_next_run = asyncio.run(
                        update_instructions(
                            llm_name=llm_name,
                            previous_instructions=instructions_for_next_run,
                            run_results=current_run_results,
                            complete_model_interaction=complete_model_interaction,
                            run_number=run_number,
                            n_runs=n_runs,
                            previous_runs=previous_models_data,
                            num_previous_runs_to_include=run_config["prompt_updater"][
                                "num_previous_runs_to_include"
                            ],
                        )
                    )
                    with open(
                        run_dir / "generated_instructions_for_next_run.txt", "w"
                    ) as f:
                        f.write(instructions_for_next_run)
                except RuntimeError as e:
                    logging.error(
                        f"RuntimeError during instruction update for run {run_number + 1}: {e}",
                        exc_info=True,
                    )
                    logging.warning(
                        "Halting execution due to instruction update error."
                    )
                    raise
                except Exception as e:
                    logging.error(
                        f"Exception during instruction update for run {run_number + 1}: {e}",
                        exc_info=True,
                    )
                    logging.warning(
                        "Halting execution due to instruction update error."
                    )
                    raise
            elif i < n_runs - 1:
                logging.info(f"Skipping instruction update for run {run_number + 1}.")

        except Exception as e:
            logging.error(f"Critical error in run {run_number}: {e}", exc_info=True)
            results_list.append(
                {
                    "run_number": run_number,
                    "average_bic": None,
                    "model_specification": f"Error: {e}",
                    "version": version,
                }
            )
            progress_bar.set_postfix_str(f"Run {run_number}: Error", refresh=True)
            print(f"\nRun {run_number} failed - Error: {e}")
            logging.warning(
                f"Skipping instruction update for run {run_number + 1} due to error."
            )

    # --- Save Final Results ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(
            "average_bic", ascending=True, na_position="last"
        )
        csv_path = exp_dir / run_config["storage"]["file_formats"]["results"]
        results_df.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
        print(f"\nExperiment complete. Results saved to {csv_path}")

        # Print summary of best run
        if not results_df.empty and results_df["average_bic"].notna().any():
            best_run = results_df.iloc[0]
            bic_str = (
                f"{best_run['average_bic']:.2f}"
                if pd.notna(best_run["average_bic"])
                else "N/A"
            )
            print(f"Best model (Run {best_run['run_number']}) - BIC: {bic_str}")
        else:
            print("No valid BIC results found across runs.")
        return results_df
    else:
        logging.error("No results were generated.")
        print("\nExperiment finished, but no results were generated.")
        return None


if __name__ == "__main__":
    run_multiple_evaluations()
