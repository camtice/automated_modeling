from inspect_ai import eval, Task
from initial_generation import (
    design_model,
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
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
import traceback


def design_model_local(
    task_description: str = TASK_DESCRIPTION,
    output_description: str = MODEL_OUTPUT_DESCRIPTION,
    dataset_path: str = CONFIG["paths"]["dataset"],
    config: Optional[dict] = None,
) -> Task:
    """
    Local version of design_model that doesn't use Docker.

    Args:
        task_description: Detailed description of the task/problem to solve
        output_description: Description of desired model outputs and characteristics
        dataset_path: Path to CSV file containing the dataset structure
        config: Configuration dictionary (if None, uses CONFIG from initial_generation)
    """
    # Use provided config or fall back to CONFIG from initial_generation
    if config is None:
        config = CONFIG

    # Get sandbox type from config, default to "local" if not specified
    sandbox_type = config.get("sandbox", {}).get("type", "local")

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
        # Use sandbox type from config
        sandbox=sandbox_type,
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
            system_message(config["system"]["prompt"]),  # Use system prompt from config
            model_design_solver(),
            simulate_model_solver(),
            parameter_fitting_solver(),
            parameter_recovery_solver(),
            bic_solver(),
            final_model_summary_solver(),
        ],
        scorer=verify(),
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file with fallback to default config."""
    base_dir = Path(__file__).parent

    if not config_path:
        config_path = base_dir / "configs" / "multiple_runs_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Store the actual config path used for later copying
    config["_config_path"] = str(config_path)

    return config


def setup_logging(log_file: Path) -> None:
    """Set up logging configuration to suppress Docker-related output."""
    # Configure root logger to only log to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    # Suppress all loggers that might output Docker-related messages
    for logger_name in [
        "inspect_ai",
        "urllib3",
        "matplotlib",
        "docker",
        "compose",
        "SANDBOX",
        "inspect_ai.util._sandbox",
        "inspect_ai.util._sandbox.docker",
        "inspect_ai.util._sandbox.docker.util",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)  # Only show errors
        logger.propagate = False  # Don't propagate to parent loggers

        # Remove any existing handlers and add file handler only
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.addHandler(logging.FileHandler(log_file))


def setup_experiment_directory(config: dict) -> tuple[Path, str]:
    """Set up experiment directory structure and return paths."""
    # Create experiment base name without timestamp
    exp_base_name = config["experiment"]["base_name"]

    # Create experiment directory under root
    root_dir = Path(config["storage"]["root_dir"])

    # Find the next available version number
    version = 0
    while True:
        exp_dir = root_dir / f"{exp_base_name}_v{version}"
        if not exp_dir.exists():
            break
        version += 1

    # Create directory structure
    dirs = {
        "data": exp_dir / config["storage"]["structure"]["data"],
        "data_raw": exp_dir / config["storage"]["structure"]["data"] / "raw",
        "configs": exp_dir / "configs",  # Add configs directory
    }

    # Create prompts directory within configs
    prompts_dir = dirs["configs"] / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = exp_dir / "experiment.log"
    setup_logging(log_file)

    # Log basic information to file
    logging.info(f"Starting experiment in directory: {exp_dir}")

    # Load environment variables from .env file
    base_dir = Path(__file__).parent
    env_path = (
        base_dir.parent / ".env"
    )  # Look in project root instead of modeling directory
    load_dotenv(env_path)

    # Get environment variable from .env and log the path we're checking
    logging.info(f"Looking for .env file at: {env_path}")
    inspect_eval_model = os.getenv("INSPECT_EVAL_MODEL", "model not set in .env file")
    logging.info(f"INSPECT_EVAL_MODEL environment variable: {inspect_eval_model}")

    # Save experiment metadata
    metadata = {
        "version": f"v{version}",
        "config": config,
        "datetime": datetime.now().isoformat(),
        "environment": {"INSPECT_EVAL_MODEL": inspect_eval_model},
    }

    with open(exp_dir / config["storage"]["file_formats"]["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy config files to the experiment directory
    base_dir = Path(__file__).parent

    # Copy multiple runs config
    if "_config_path" in config:
        config_path = Path(config["_config_path"])
        if config_path.exists():
            shutil.copy2(config_path, dirs["configs"] / config_path.name)

    # Copy default config
    default_config_path = base_dir / "configs" / "default_config.yaml"
    if default_config_path.exists():
        shutil.copy2(default_config_path, dirs["configs"] / "default_config.yaml")

    # Copy prompt files
    prompt_files = ["model_output_description.txt", "task_description.txt"]
    for prompt_file in prompt_files:
        source_path = base_dir / "configs" / "prompts" / prompt_file
        if source_path.exists():
            shutil.copy2(source_path, prompts_dir / prompt_file)

    return exp_dir, f"v{version}"


def run_multiple_evaluations(
    n_runs: Optional[int] = None, config_path: Optional[str] = None
):
    """Run multiple evaluations and store results."""
    # Load configuration
    config = load_config(config_path)

    # Update n_runs in the experiment section if provided
    if n_runs is not None:
        config["experiment"]["n_runs"] = n_runs
    elif "n_runs" not in config["experiment"]:
        raise ValueError("Number of runs (n_runs) must be specified either in config")

    # Setup experiment directory
    exp_dir, version = setup_experiment_directory(config)

    # Print minimal info to console
    print(f"Experiment {version} started in: {exp_dir}")
    print(f"Running {config['experiment']['n_runs']} evaluations...")

    # Load default config for model settings
    base_dir = Path(__file__).parent
    default_config_path = base_dir / "configs" / "default_config.yaml"
    with open(default_config_path, "r") as f:
        default_config = yaml.safe_load(f)

    # Log sandbox type being used
    sandbox_type = default_config.get("sandbox", {}).get("type", "local")
    logging.info(f"Using sandbox type: {sandbox_type}")
    print(f"Using sandbox type: {sandbox_type}")

    results_list = []
    previous_models = []  # Track previous models

    # Use tqdm for progress bar
    progress_bar = tqdm(
        range(config["experiment"]["n_runs"]), desc="Running evaluations", unit="run"
    )

    for i in progress_bar:
        # Log to file only
        logging.info(f"Running evaluation {i + 1}/{config['experiment']['n_runs']}")

        try:
            # Create directory for this run
            run_dir = exp_dir / "data" / "raw" / f"run_{i + 1}"
            plots_dir = run_dir / config["storage"]["structure"]["plots"]
            run_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Update model output description with previous models if enabled
            if (
                config["experiment"].get("consider_previous_models", True)
                and previous_models
            ):
                previous_models_text = "Previous Models:\n"
                for idx, model in enumerate(previous_models, 1):
                    previous_models_text += f"Model {idx}:\n"
                    previous_models_text += f"Specification: {model['specification']}\n"
                    previous_models_text += f"Summary: {model['summary']}\n"
                    if (
                        config["experiment"].get("include_bic_in_summary", False)
                        and model.get("bic") is not None
                    ):
                        previous_models_text += f"BIC: {model['bic']}\n"
                    # Add recovery information if available
                    if model.get("recovery_info") and config["experiment"].get(
                        "include_recovery_in_summary", True
                    ):
                        previous_models_text += f"{model['recovery_info']}\n"

                enhanced_output_description = (
                    MODEL_OUTPUT_DESCRIPTION.rstrip()
                    + "\n\n"
                    + config["experiment"].get("previous_models_prompt", "").strip()
                    + "\n\n"
                    + previous_models_text.rstrip()
                )
            else:
                enhanced_output_description = MODEL_OUTPUT_DESCRIPTION

            # Run evaluation with enhanced description
            # Set environment variable to disable Docker output
            os.environ["INSPECT_QUIET"] = "1"

            # Run the evaluation using the configurable version of design_model
            log_list = eval(
                design_model_local(
                    output_description=enhanced_output_description,
                    config=default_config,
                ),
                trace=False,
            )
            log = log_list[0]  # Single sample task

            if log.samples and log.samples[0].metadata:
                md = log.samples[0].metadata

                # Extract results
                # BIC values are stored directly in the metadata, not inside a "bic_results" dictionary
                bic_value = md.get("average_bic", None)
                bic_control = md.get("bic_Control", None)
                bic_cocaine = md.get("bic_Cocaine", None)

                model_spec = md.get("model_specification", "")
                if not model_spec:
                    logging.warning(f"No model specification found in metadata: {md}")

                model_summary = md.get("model_summary", "")
                if not model_summary:
                    logging.warning(f"No model summary found in metadata: {md}")

                recovery_info = md.get("parameter_recovery", {}).get("correlations", {})
                if not recovery_info:
                    logging.warning(
                        f"No parameter recovery info found in metadata: {md}"
                    )

                # Track this model for future runs
                previous_models.append(
                    {
                        "specification": model_spec,
                        "summary": model_summary,
                        "bic": bic_value,
                        "recovery_info": md.get("recovery_summary", ""),
                    }
                )

                # Move parameter recovery plots if they exist
                if os.path.exists("param_recovery_plots"):
                    for file in os.listdir("param_recovery_plots"):
                        shutil.move(
                            f"param_recovery_plots/{file}", str(plots_dir / file)
                        )
                    os.rmdir("param_recovery_plots")

                # Create row for results
                row = {
                    "run_number": i + 1,
                    "average_bic": bic_value,
                    "bic_control": bic_control,
                    "bic_cocaine": bic_cocaine,
                    "model_specification": model_spec,
                    "model_summary": model_summary,
                    "version": version,
                }

                # Add parameter recovery correlations
                recovery_values = {}
                for param, data in recovery_info.items():
                    r_value = data.get("r", None)
                    row[f"{param}_recovery"] = r_value
                    recovery_values[param] = r_value

                results_list.append(row)

                # Save individual run metadata
                with open(run_dir / "metadata.json", "w") as f:
                    json.dump(md, f, indent=2)

                # Update progress bar description with current model info
                recovery_str = ", ".join(
                    [
                        f"{p}={v:.2f}" if v is not None else f"{p}=None"
                        for p, v in recovery_values.items()
                    ]
                )

                # Format BIC values with proper handling of None values
                if bic_value is not None:
                    bic_str = f"BIC={bic_value:.2f}"
                else:
                    bic_str = "BIC=None"

                if bic_control is not None:
                    bic_str += f", Control={bic_control:.2f}"
                if bic_cocaine is not None:
                    bic_str += f", Cocaine={bic_cocaine:.2f}"

                model_info = f"Run {i + 1}: {bic_str}, Recovery: {recovery_str}"
                progress_bar.set_postfix_str(model_info)

                # Also print a summary line after each run
                print(
                    f"Run {i + 1} complete - {bic_str}, Parameter recovery: {recovery_str}"
                )

            else:
                logging.warning(f"No metadata found for run {i + 1}")
                results_list.append(
                    {
                        "run_number": i + 1,
                        "average_bic": None,
                        "model_specification": "No metadata found",
                        "model_summary": "No metadata found",
                        "version": version,
                    }
                )
                # Update progress bar with error info
                progress_bar.set_postfix_str(f"Run {i + 1}: No metadata found")
                print(f"Run {i + 1} complete - No metadata found")

        except Exception as e:
            # Log the full exception with traceback
            logging.error(f"Error in run {i + 1}: {str(e)}")
            logging.error(traceback.format_exc())

            # Add a placeholder for BIC and recovery values in the results
            error_row = {
                "run_number": i + 1,
                "average_bic": None,
                "model_specification": f"Error: {str(e)}",
                "model_summary": f"Error: {str(e)}",
                "version": version,
            }
            results_list.append(error_row)

            # Update progress bar with error info
            progress_bar.set_postfix_str(f"Run {i + 1}: Error occurred")
            print(f"Run {i + 1} complete - Error: {str(e)}")

    # Save results to CSV at the experiment root level
    results_df = pd.DataFrame(results_list)
    # Sort by average_bic from lowest to highest
    results_df = results_df.sort_values("average_bic", ascending=True)
    csv_path = exp_dir / config["storage"]["file_formats"]["results"]
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")

    # Print final summary
    print(f"\nExperiment complete. Results saved to {csv_path}")
    if not results_df.empty and results_df["average_bic"].notna().any():
        best_model = results_df.iloc[0]
        if best_model["average_bic"] is not None:
            print(
                f"Best model (Run {best_model['run_number']}) - BIC: {best_model['average_bic']:.2f}"
            )
        else:
            print(f"Best model (Run {best_model['run_number']}) - BIC: None")

    return results_df


if __name__ == "__main__":
    # Run evaluations
    df = run_multiple_evaluations()
