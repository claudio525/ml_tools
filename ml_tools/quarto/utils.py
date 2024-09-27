from pathlib import Path
import shutil
import subprocess

def render_quarto(
    activate_env_cmd: str,
    notebook_ffp: Path,
    output_ffp: Path,
    output_format: str = "html",
    **kwargs,
):
    """
    Runs the quarto render command to render a jupyter notebook

    Parameters
    ----------
    activate_env_cmd: str
        Command to activate the environment
    notebook_ffp: Path
        Path to the notebook file
    output_ffp: Path
        Path to the output file
    output_format: str
        Output format of the file
    kwargs: dict
        Additional arguments to pass to quarto.
        For each key-value pair, the following is
        added to the command: -P key:value
    """

    # Define the quarto command
    quarto_command = (
        f"quarto render {notebook_ffp} --to {output_format} --execute"
    )

    if kwargs is not None:
        for key, value in kwargs.items():
            quarto_command += f" -P {key}:{value}"

    command = f"source ~/.zshrc && {activate_env_cmd} && {quarto_command}"
    try:
        # Run the command
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="/bin/zsh",
        )

        # Output the result
        print("Render successful!")
        print(result.stdout.decode())

        # Move the output file to the desired location
        result_ffp = notebook_ffp.parent / Path(notebook_ffp.stem + f".{output_format}")
        shutil.move(result_ffp, output_ffp)

    except subprocess.CalledProcessError as e:
        print(f"Error during rendering: {e.stderr.decode()}")
