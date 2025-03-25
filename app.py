#!/usr/bin/env python3
from flask import Flask, render_template, request, url_for, send_from_directory, abort
import pandas as pd
import json
import os
import yaml
from pathlib import Path
from werkzeug.exceptions import HTTPException
import time

app = Flask(__name__)


# Add YAML filter
@app.template_filter("yaml")
def yaml_filter(data):
    return yaml.dump(
        data, sort_keys=False, default_flow_style=False, allow_unicode=True
    )


# Add timestamp_to_datetime filter
@app.template_filter("timestamp_to_datetime")
def timestamp_to_datetime(timestamp):
    """Convert a Unix timestamp to a readable datetime format."""
    if timestamp is None:
        return "N/A"
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


# Add safe float format filter
@app.template_filter("safe_float")
def safe_float_filter(value):
    """Safely format a float value, returning 'N/A' for None or non-numeric values."""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        return "{:.2f}".format(float(value))
    except (ValueError, TypeError):
        return "N/A"


# Define experiments directory
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.getcwd(), "modeling", "experiments"))
CONFIGS_DIR = os.path.abspath(os.path.join(os.getcwd(), "modeling", "configs"))

# Function to find the most recent experiment with results.csv
def find_most_recent_experiment():
    """Find the most recent experiment directory that contains a results.csv file."""
    fallback = "testing_v1"  # Default fallback if no valid experiment is found
    
    if not os.path.exists(EXPERIMENTS_DIR):
        return fallback
    
    def get_creation_time(path):
        return os.path.getctime(path)
    
    # Function to recursively find directories with results.csv
    def find_results_dirs(base_dir, max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return []
        
        valid_dirs = []
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    results_path = os.path.join(item_path, "results.csv")
                    if os.path.exists(results_path):
                        valid_dirs.append(item_path)
                    # Recursively search in subdirectories
                    valid_dirs.extend(find_results_dirs(item_path, max_depth, current_depth + 1))
        except Exception as e:
            print(f"Error scanning directory {base_dir}: {e}")
        
        return valid_dirs
    
    # Find all directories with results.csv
    valid_experiment_dirs = find_results_dirs(EXPERIMENTS_DIR)
    
    if not valid_experiment_dirs:
        print("Warning: No valid experiment directories found with results.csv")
        return fallback
    
    # Sort by creation time (newest first) and get the most recent
    valid_experiment_dirs.sort(key=get_creation_time, reverse=True)
    most_recent_dir = valid_experiment_dirs[0]
    
    # Get the relative path from EXPERIMENTS_DIR
    rel_path = os.path.relpath(most_recent_dir, EXPERIMENTS_DIR)
    print(f"Using most recent experiment: {rel_path}")
    
    return rel_path

# Set default experiment to the most recent valid one
DEFAULT_EXPERIMENT = find_most_recent_experiment()

def validate_experiment_dir():
    """Validate that the experiments directory exists."""
    if not os.path.exists(EXPERIMENTS_DIR):
        raise RuntimeError(f"Experiments directory not found at: {EXPERIMENTS_DIR}")


def load_experiment_metadata(exp_version):
    """Load the results.csv and metadata.json for a given experiment version."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_version)
    if not os.path.exists(exp_dir):
        abort(404, description=f"Experiment version {exp_version} not found")

    results_path = os.path.join(exp_dir, "results.csv")
    metadata_path = os.path.join(exp_dir, "metadata.json")
    config_path = os.path.join(exp_dir, "configs", "default_config.yaml")

    df = pd.DataFrame()
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        # Convert any missing values to None for consistent handling
        df = df.replace({pd.NA: None})
    else:
        abort(404, description=f"Results file not found for experiment {exp_version}")

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Load experiment config to check if group analysis is enabled
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Add group analysis data if enabled
    if config.get("group_analysis", {}).get("enabled", False):
        group_column = config["group_analysis"]["group_column"]
        base_data_path = os.path.join(exp_dir, "data", "raw")

        # Load the original dataset to get group information
        dataset_path = os.path.join(os.getcwd(), "modeling", config["paths"]["dataset"])
        if os.path.exists(dataset_path):
            base_df = pd.read_csv(dataset_path)
            unique_groups = base_df[group_column].unique()

            # Add group information to metadata
            metadata["group_analysis"] = {"groups": list(unique_groups), "metrics": {}}

            # Process each run's group-specific metrics and metadata
            for run_number in df["run_number"]:
                # Load run-specific metadata first
                run_metadata_path = os.path.join(
                    base_data_path, f"run_{run_number}", "metadata.json"
                )
                if os.path.exists(run_metadata_path):
                    with open(run_metadata_path) as f:
                        run_metadata = json.load(f)
                        # Add BIC scores from metadata to DataFrame
                        if "average_bic" in run_metadata:
                            df.loc[df["run_number"] == run_number, "average_bic"] = (
                                run_metadata["average_bic"]
                            )
                        for key in run_metadata:
                            if key.startswith("bic_"):
                                df.loc[df["run_number"] == run_number, key] = (
                                    run_metadata[key]
                                )

                # Then process group metrics
                run_metrics = {"bic": {}, "parameters": {}}
                run_path = os.path.join(
                    base_data_path, f"run_{run_number}", "group_metrics.json"
                )

                if os.path.exists(run_path):
                    with open(run_path) as f:
                        run_metrics = json.load(f)

                # Add group metrics to the main DataFrame
                for group in unique_groups:
                    if "parameters" in run_metrics:
                        for param, values in run_metrics["parameters"].items():
                            if group in values:
                                recovery_value = values.get(group)
                                df.loc[
                                    df["run_number"] == run_number,
                                    f"{param}_recovery_{group}",
                                ] = (
                                    recovery_value
                                    if recovery_value is not None
                                    else None
                                )

    return df, metadata


def get_plot_paths(exp_version, run_number):
    """Get all parameter recovery plots for a specific run."""
    plots_dir = os.path.join(
        EXPERIMENTS_DIR, exp_version, "data", "raw", f"run_{run_number}", "plots"
    )
    plot_files = []

    if os.path.exists(plots_dir):
        for file in os.listdir(plots_dir):
            if file.endswith(".png") and file.startswith("parameter_recovery_"):
                plot_files.append(file)

    return plots_dir, plot_files


def load_yaml_config(filepath):
    """Load a YAML configuration file."""
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}


def get_experiment_configs(exp_version):
    """Get all configuration files for an experiment."""
    # Load the default config
    default_config = load_yaml_config(os.path.join(CONFIGS_DIR, "default_config.yaml"))

    # Load the multiple runs config
    multiple_runs_config = load_yaml_config(
        os.path.join(CONFIGS_DIR, "multiple_runs_config.yaml")
    )

    # Load experiment-specific metadata
    exp_metadata = {}
    metadata_path = os.path.join(EXPERIMENTS_DIR, exp_version, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            exp_metadata = json.load(f)

    return {
        "default_config": default_config,
        "multiple_runs_config": multiple_runs_config,
        "experiment_metadata": exp_metadata,
    }


def get_all_experiments():
    """Recursively get all experiment directories and organize them by date and type."""
    experiments = []

    # Dictionary to organize experiments by date
    organized_experiments = {}

    for root, dirs, files in os.walk(EXPERIMENTS_DIR):
        # Skip the root experiments directory itself
        if root == EXPERIMENTS_DIR:
            continue

        # Get relative path from EXPERIMENTS_DIR
        rel_path = os.path.relpath(root, EXPERIMENTS_DIR)

        # Check if this is an experiment directory (contains results.csv)
        if "results.csv" in files:
            # Parse the path to extract date and experiment name
            path_parts = rel_path.split("/")

            # If the path has a date component (like feb_4)
            date_prefix = None
            experiment_name = rel_path

            # Check if the first part looks like a date (month_day format)
            if len(path_parts) > 1 and "_" in path_parts[0]:
                date_parts = path_parts[0].split("_")
                if (
                    len(date_parts) == 2
                    and date_parts[0].isalpha()
                    and date_parts[1].isdigit()
                ):
                    date_prefix = path_parts[0]
                    experiment_name = "/".join(path_parts[1:])

            # Group by date if available, otherwise use "other"
            group = date_prefix if date_prefix else "other"

            if group not in organized_experiments:
                organized_experiments[group] = []

            organized_experiments[group].append(
                {"path": rel_path, "name": experiment_name, "full_path": rel_path}
            )

    # Sort groups by date (assuming format like feb_4, mar_10, etc.)
    # First, define month order
    month_order = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    # Sort groups
    sorted_groups = sorted(
        organized_experiments.keys(),
        key=lambda x: (
            # If it's a month_day format, sort by month then day
            (month_order.get(x.split("_")[0], 13), int(x.split("_")[1]))
            if "_" in x and x.split("_")[0] in month_order and x.split("_")[1].isdigit()
            else (99, 99)  # Put "other" at the end
        ),
        reverse=True,  # Most recent dates first
    )

    # Build the final sorted list
    for group in sorted_groups:
        # Sort experiments within each group alphabetically
        sorted_exps = sorted(organized_experiments[group], key=lambda x: x["name"])
        for exp in sorted_exps:
            experiments.append(exp)

    return experiments, sorted_groups


@app.route("/")
def index():
    # Get experiment version from query string
    exp_version = request.args.get("experiment", DEFAULT_EXPERIMENT)

    try:
        # Load experiment data
        models_df, metadata = load_experiment_metadata(exp_version)

        # Get list of available experiments
        experiments, sorted_groups = get_all_experiments()

        model_list = models_df.to_dict(orient="records") if not models_df.empty else []

        return render_template(
            "index.html",
            exp_version=exp_version,
            experiments=experiments,
            models=model_list,
            metadata=metadata,
            sorted_groups=sorted_groups,
        )
    except Exception as e:
        abort(500, description=str(e))


@app.route("/model/<int:run_number>")
def model_detail(run_number):
    exp_version = request.args.get("experiment", DEFAULT_EXPERIMENT)

    try:
        models_df, metadata = load_experiment_metadata(exp_version)

        # Get model info
        model_info = models_df[models_df["run_number"] == run_number].to_dict(
            orient="records"
        )
        if not model_info:
            abort(404, description=f"Model run {run_number} not found")

        model_info = model_info[0]

        # Get plot paths
        plots_dir, plot_files = get_plot_paths(exp_version, run_number)

        # Load run-specific metadata
        run_metadata_path = os.path.join(
            EXPERIMENTS_DIR,
            exp_version,
            "data",
            "raw",
            f"run_{run_number}",
            "metadata.json",
        )

        if os.path.exists(run_metadata_path):
            with open(run_metadata_path) as f:
                run_metadata = json.load(f)
                # Add run-specific metadata to the general metadata
                metadata.update(run_metadata)

        return render_template(
            "model_detail.html",
            exp_version=exp_version,
            model_info=model_info,
            metadata=metadata,
            plot_files=plot_files,
            run_number=run_number,
        )
    except Exception as e:
        abort(500, description=str(e))


@app.route("/plots/<path:exp_version>/run_<int:run_number>/<path:filename>")
def serve_plot(exp_version, run_number, filename):
    plots_dir = os.path.join(
        EXPERIMENTS_DIR, exp_version, "data", "raw", f"run_{run_number}", "plots"
    )
    if not os.path.exists(plots_dir):
        abort(404, description=f"Plot directory not found for run {run_number}")

    return send_from_directory(plots_dir, filename)


@app.route("/config/<exp_version>")
def config_page(exp_version):
    """Display configuration information for an experiment."""
    try:
        configs = get_experiment_configs(exp_version)
        experiments, sorted_groups = get_all_experiments()

        return render_template(
            "config.html",
            exp_version=exp_version,
            experiments=experiments,
            configs=configs,
            sorted_groups=sorted_groups,
        )
    except Exception as e:
        abort(500, description=str(e))


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Handle all HTTP exceptions."""
    return render_template(
        "error.html", error_code=e.code, error_description=e.description
    ), e.code


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle non-HTTP exceptions."""
    return render_template("error.html", error_code=500, error_description=str(e)), 500


# Validate experiments directory on startup
validate_experiment_dir()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
