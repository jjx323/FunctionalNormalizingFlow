import os
import subprocess
import sys
import time


def run_script(script_name, description=""):
    """Execute specified Python script"""
    print(f"\n{'=' * 60}")
    print(f"Starting: {script_name}")
    if description:
        print(f"Description: {description}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Execute the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )

        # Print output
        if result.stdout:
            print("Output:")
            print(result.stdout)

        elapsed_time = time.time() - start_time
        print(f"\n✓ {script_name} executed successfully!")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {script_name} execution failed!")
        print(f"Error code: {e.returncode}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        if e.stdout:
            print("Standard output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)

        # Ask user whether to continue
        response = input("\nEncountered error, continue with remaining scripts? (y/n): ")
        if response.lower() != 'y':
            print("Program terminated.")
            sys.exit(1)

    return True


def check_file_exists(script_name):
    """Check if file exists"""
    if not os.path.exists(script_name):
        print(f"Warning: File {script_name} does not exist!")
        return False
    return True


def main():
    """Main function: execute all scripts in sequence"""
    print("=" * 60)
    print("Starting functional normalizing flow experiment pipeline")
    print("=" * 60)

    # Define scripts to run and their descriptions
    scripts = [
        ("generate_eig.py", "Generate eigenfunctions of prior measure"),
        ("generate_data.py", "Generate real function and measurement data with noise"),
        ("experiment.py", "Train functional normalizing flow"),
        ("model_plot.py", "Plot results of functional normalizing flow"),
        ("discrete_invariance.py", "Run discrete invariance experiment")
    ]

    # Check if all required files exist
    print("Checking required files...")
    missing_files = []
    for script_name, _ in scripts:
        if not check_file_exists(script_name):
            missing_files.append(script_name)

    if missing_files:
        print(f"\nMissing files: {missing_files}")
        response = input("Continue execution? (y/n): ")
        if response.lower() != 'y':
            print("Program terminated.")
            sys.exit(1)

    # Record start time
    total_start_time = time.time()

    # Execute all scripts in sequence
    successful_runs = 0
    failed_runs = []

    for script_name, description in scripts:
        if os.path.exists(script_name):
            try:
                run_script(script_name, description)
                successful_runs += 1
            except KeyboardInterrupt:
                print("\nUser interrupted execution.")
                sys.exit(0)
            except Exception as e:
                print(f"Unknown error while running {script_name}: {e}")
                failed_runs.append(script_name)
        else:
            print(f"\nSkipping {script_name} (file does not exist)")
            failed_runs.append(script_name)

    # Statistics
    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successfully executed: {successful_runs}/{len(scripts)} scripts")

    if failed_runs:
        print(f"Failed scripts: {failed_runs}")
    else:
        print("✓ All scripts executed successfully!")

    # Create execution report
    report_file = "run_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Execution time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Successfully executed: {successful_runs}/{len(scripts)} scripts\n")
        if failed_runs:
            f.write(f"Failed scripts: {failed_runs}\n")

    print(f"\nExecution report saved to: {report_file}")


if __name__ == "__main__":
    main()