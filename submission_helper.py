"""
IML26 Assignment 2 - Submission Helper
=======================================
Use this module to save your trained models and create your final submission zip.

IMPORTANT: You MUST use these functions to save your models.
The grader will only accept files saved with these helpers.
"""

import os
import json
import zipfile
import tempfile
import shutil
from datetime import datetime


def save_task1_model(model, filepath: str) -> None:
    """
    Save your PyTorch image classification model for Task 1.

    Parameters
    ----------
    model : torch.nn.Module
        Your trained PyTorch model. Must be callable with input shape (N, 3, 32, 32)
        and return logits of shape (N, 2).
    filepath : str
        Destination path, e.g. 'task1_model.pkl'

    Example
    -------
    >>> from submission_helper import save_task1_model
    >>> save_task1_model(my_model, 'task1_model.pkl')
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required: pip install torch")

    import io
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    # Compile to TorchScript so the grader can load any custom architecture
    # without needing the original class definition.
    scripted_bytes = None
    try:
        scripted = torch.jit.script(model)
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        scripted_bytes = buf.getvalue()
    except Exception:
        try:
            example = torch.zeros(1, 3, 32, 32)
            traced = torch.jit.trace(model, example)
            buf = io.BytesIO()
            torch.jit.save(traced, buf)
            scripted_bytes = buf.getvalue()
        except Exception:
            pass  # fallback: grader will attempt state_dict reconstruction

    payload = {
        "task": "task1",
        "scripted_bytes": scripted_bytes,        # self-contained; no class needed at load
        "model_state_dict": model.state_dict(),  # kept for inspection / fallback
        "model_class": model.__class__.__name__,
        "total_params": total_params,
        "saved_at": datetime.utcnow().isoformat(),
    }
    torch.save(payload, filepath)
    method = "TorchScript" if scripted_bytes else "state_dict (fallback)"
    print(f"[Task 1] Model saved to '{filepath}' | Parameters: {total_params:,} | Format: {method}")


def save_task2_model(pipeline, filepath: str) -> None:
    """
    Save your scikit-learn pipeline for Task 2.

    Parameters
    ----------
    pipeline : sklearn estimator / Pipeline
        Fitted sklearn object with a .predict(X) method that accepts
        a numpy array of shape (N, 2000) and returns class labels.
    filepath : str
        Destination path, e.g. 'task2_model.pkl'

    Example
    -------
    >>> from submission_helper import save_task2_model
    >>> save_task2_model(my_pipeline, 'task2_model.pkl')
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required: pip install joblib")

    payload = {
        "task": "task2",
        "pipeline": pipeline,
        "saved_at": datetime.utcnow().isoformat(),
    }
    joblib.dump(payload, filepath)
    print(f"[Task 2] Pipeline saved to '{filepath}'")


def save_task3_model(model, filepath: str) -> None:
    """
    Save your constrained PyTorch model for Task 3.

    CONSTRAINT: Your model must have AT MOST 500 trainable parameters.
    The grader will automatically reject models with more than 500 parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Your trained PyTorch model. Must accept input shape (N, 20) and return
        logits of shape (N, 1).
    filepath : str
        Destination path, e.g. 'task3_model.pkl'

    Raises
    ------
    ValueError
        If the model has more than 500 trainable parameters.

    Example
    -------
    >>> from submission_helper import save_task3_model
    >>> save_task3_model(my_model, 'task3_model.pkl')
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required: pip install torch")

    import io
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())

    if total_params > 500:
        raise ValueError(
            f"Model has {total_params} parameters, but the limit is 500. "
            "Reduce your model size before saving."
        )

    # Compile to TorchScript so the grader can load any custom architecture
    # without needing the original class definition.
    scripted_bytes = None
    try:
        scripted = torch.jit.script(model)
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        scripted_bytes = buf.getvalue()
    except Exception:
        try:
            example = torch.zeros(1, 20)
            traced = torch.jit.trace(model, example)
            buf = io.BytesIO()
            torch.jit.save(traced, buf)
            scripted_bytes = buf.getvalue()
        except Exception:
            pass  # fallback: grader will attempt state_dict reconstruction

    payload = {
        "task": "task3",
        "scripted_bytes": scripted_bytes,        # self-contained; no class needed at load
        "model_state_dict": model.state_dict(),  # kept for inspection / fallback
        "model_class": model.__class__.__name__,
        "total_params": total_params,
        "saved_at": datetime.utcnow().isoformat(),
    }
    torch.save(payload, filepath)
    method = "TorchScript" if scripted_bytes else "state_dict (fallback)"
    print(f"[Task 3] Model saved to '{filepath}' | Parameters: {total_params} / 500 (VALID) | Format: {method}")


def _add_code_folder_to_zip(zf: zipfile.ZipFile, code_dir: str, arc_root: str = "code") -> int:
    """
    Recursively add source files from code_dir into the zip under arc_root/.

    Includes: *.py, requirements.txt, environment.yml
    Skips: __pycache__, .git, venv, .venv, env, node_modules
    """
    if not os.path.isdir(code_dir):
        raise FileNotFoundError(
            f"Required folder '{code_dir}' not found. Create it and copy all Python "
            f"source files you used to train your models (see assignment PDF)."
        )

    code_dir = os.path.abspath(code_dir)
    skip_dir_names = {"__pycache__", ".git", ".venv", "venv", "env", "node_modules"}
    optional_files = {"requirements.txt", "environment.yml"}
    count = 0

    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in skip_dir_names]
        for fname in files:
            lower = fname.lower()
            if lower.endswith(".py") or fname in optional_files:
                full_path = os.path.join(root, fname)
                rel = os.path.relpath(full_path, code_dir)
                arcname = os.path.join(arc_root, rel).replace("\\", "/")
                zf.write(full_path, arcname)
                count += 1

    if count == 0:
        raise ValueError(
            f"No .py files (or requirements.txt / environment.yml) found in '{code_dir}'."
        )
    return count


def create_submission(
    student_email: str,
    task1_pkl: str,
    task2_pkl: str,
    task3_pkl: str,
    output_zip: str = "submission.zip",
    code_dir: str = "code",
) -> None:
    """
    Bundle all three model files and your source-code folder into a single submission zip.

    Parameters
    ----------
    student_email : str
        Your university email address (used for grading identification).
    task1_pkl : str
        Path to the file saved by save_task1_model().
    task2_pkl : str
        Path to the file saved by save_task2_model().
    task3_pkl : str
        Path to the file saved by save_task3_model().
    output_zip : str
        Output zip filename (default: 'submission.zip').
    code_dir : str
        Folder containing all .py scripts you wrote for this assignment (and
        optionally requirements.txt or environment.yml). Stored in the zip under
        ``code/`` for plagiarism review. Default ``\"code\"``.

    Example
    -------
    >>> from submission_helper import create_submission
    >>> create_submission(
    ...     student_email='john.doe@university.edu',
    ...     task1_pkl='task1_model.pkl',
    ...     task2_pkl='task2_model.pkl',
    ...     task3_pkl='task3_model.pkl',
    ...     output_zip='submission.zip',
    ...     code_dir='code',
    ... )
    """
    for path, label in [(task1_pkl, "task1_pkl"), (task2_pkl, "task2_pkl"), (task3_pkl, "task3_pkl")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[{label}] File not found: '{path}'. Run the corresponding save function first.")

    if not student_email or "@" not in student_email:
        raise ValueError(f"Invalid email address: '{student_email}'")

    manifest = {
        "student_email": student_email.strip().lower(),
        "submission_timestamp": datetime.utcnow().isoformat(),
        "files": {
            "task1": os.path.basename(task1_pkl),
            "task2": os.path.basename(task2_pkl),
            "task3": os.path.basename(task3_pkl),
        },
        "code_directory": "code",
    }

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.write(task1_pkl, os.path.basename(task1_pkl))
        zf.write(task2_pkl, os.path.basename(task2_pkl))
        zf.write(task3_pkl, os.path.basename(task3_pkl))
        n_code = _add_code_folder_to_zip(zf, code_dir, arc_root="code")

    size_kb = os.path.getsize(output_zip) / 1024
    print(f"\n{'='*50}")
    print(f"  Submission created: '{output_zip}'")
    print(f"  Student email:      {student_email}")
    print(f"  Size:               {size_kb:.1f} KB")
    print(f"  Files included:     manifest.json, {os.path.basename(task1_pkl)}, "
          f"{os.path.basename(task2_pkl)}, {os.path.basename(task3_pkl)}")
    print(f"  Source code:        {n_code} file(s) under code/ (from '{code_dir}')")
    print(f"{'='*50}\n")
    print("Submit this zip file via the course portal.")
