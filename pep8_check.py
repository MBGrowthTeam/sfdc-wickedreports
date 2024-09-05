import subprocess
import sys
import os


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode


def is_excluded(file_path):
    excluded_paths = ["bootstrap.py", "salesforce_data_export/app/"]
    return any(excluded_path in file_path for excluded_path in excluded_paths)


def main():
    print("Running Black formatter...")
    black_command = "black ."
    stdout, stderr, returncode = run_command(black_command)
    if returncode != 0:
        print(f"Error running Black: {stderr}")
        sys.exit(1)
    print("Black formatting completed successfully.")

    print("\nRunning flake8...")
    flake8_command = "flake8"
    stdout, stderr, returncode = run_command(flake8_command)
    if stdout:
        print("flake8 found the following issues:")
        print(stdout)
    else:
        print("flake8 found no issues.")

    if returncode != 0:
        print("Please fix any remaining issues manually and run the script again.")
        sys.exit(1)
    else:
        print("PEP 8 check completed successfully.")


if __name__ == "__main__":
    main()
