#!/usr/bin/env python3
"""
Utility functions for working with Jupyter notebooks.
"""

import os
from typing import Tuple
import typer
import nbformat
from pathlib import Path


def clear_notebook_outputs(notebook_path: str) -> bool:
    """
    Clear the outputs from a Jupyter notebook.
    
    Parameters
    ----------
    notebook_path : str
        Path to the notebook file.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Clear outputs
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                if 'outputs' in cell:
                    cell.outputs = []
                if 'execution_count' in cell:
                    cell.execution_count = None
        
        # Write the notebook back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        print(f"Cleared outputs in {notebook_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False


def clear_notebooks_in_directory(directory: str, recursive: bool = False) -> Tuple[int, int]:
    """
    Clear outputs from all notebooks in the specified directory.
    
    Parameters
    ----------
    directory : str
        Directory containing notebooks.
    recursive : bool, optional
        Whether to search recursively in subdirectories, by default False.
    
    Returns
    -------
    Tuple[int, int]
        A tuple of (processed_count, success_count)
    """
    processed = 0
    success = 0
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.ipynb'):
                    processed += 1
                    if clear_notebook_outputs(os.path.join(root, file)):
                        success += 1
    else:
        for file in os.listdir(directory):
            if file.endswith('.ipynb'):
                processed += 1
                if clear_notebook_outputs(os.path.join(directory, file)):
                    success += 1
    
    return processed, success


def main(
    directory: Path = typer.Argument(..., help="Directory containing notebooks"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search recursively in subdirectories"),
) -> None:
    """
    Clear outputs from Jupyter notebooks.
    """
    if not directory.is_dir():
        typer.echo(f"Error: {directory} is not a valid directory")
        raise typer.Exit(code=1)
    
    processed, success = clear_notebooks_in_directory(str(directory), recursive)
    
    typer.echo(f"Summary: Successfully cleared {success} of {processed} notebooks")


if __name__ == "__main__":
    typer.run(main)
