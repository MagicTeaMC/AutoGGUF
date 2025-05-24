#!/usr/bin/env python3

import os
import subprocess
import sys

# Available quantization types
QUANTIZATION_TYPES = [
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_0",
    "Q4_1",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
]


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}")
    print(f"Running: {command}")
    print("-" * 50)

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=False, text=True
        )
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}")
        print(f"Command failed with return code: {e.returncode}")
        return False


def get_user_input():
    """Get user input for model path and quantization selection"""
    print("Llama.cpp Model Converter and Quantizer")
    print("=" * 50)

    # Get model path
    while True:
        model_path = input(
            "\nEnter the model directory path (e.g., 'my-model'): "
        ).strip()
        if not model_path:
            print("Model path cannot be empty")
            continue

        # Check if the path exists
        if not os.path.exists(model_path):
            print(f"Warning: Directory '{model_path}' does not exist")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != "y":
                continue

        break

    # Get output name (default to model path name)
    model_name = os.path.basename(model_path.rstrip("/"))
    output_name = input(
        f"\nEnter output filename prefix (default: '{model_name}'): "
    ).strip()
    if not output_name:
        output_name = model_name

    # Show quantization options
    print("\nQuantization options:")
    print(f"Available types: {', '.join(QUANTIZATION_TYPES)}")
    print("\nOptions:")
    print("1. Convert to ALL quantization types (default)")
    print("2. Select specific quantization types")

    # Get quantization selection
    while True:
        try:
            choice = input(
                    "\nSelect option (1-2, or press Enter for default): "
            ).strip()

            # Default to all quantization types if empty input
            if not choice or choice == "1":
                selected_quants = QUANTIZATION_TYPES.copy()
                print(
                    f"✅ Selected: ALL quantization types ({len(selected_quants)} types)"
                )
                break
            elif choice == "2":
                print("\nAvailable quantization types:")
                for i, q_type in enumerate(QUANTIZATION_TYPES, 1):
                    print(f"{i:2d}. {q_type}")

                print(
                    "\nEnter quantization types (comma-separated, e.g., 'Q4_K_M,Q5_K_M'):"
                )
                custom_input = input("Types: ").strip()

                if not custom_input:
                    print("❌ No types selected, using all types")
                    selected_quants = QUANTIZATION_TYPES.copy()
                    break

                custom_types = [t.strip().upper() for t in custom_input.split(",")]

                # Validate custom types
                invalid_types = [t for t in custom_types if t not in QUANTIZATION_TYPES]
                if invalid_types:
                    print(f"❌ Invalid quantization types: {', '.join(invalid_types)}")
                    print(f"Valid types: {', '.join(QUANTIZATION_TYPES)}")
                    continue

                selected_quants = custom_types
                print(f"✅ Selected: {', '.join(selected_quants)}")
                break
            else:
                print("❌ Please enter 1 or 2 (or press Enter for default)")
        except ValueError:
            print("❌ Please enter a valid option")

    return model_path, output_name, selected_quants


def convert_to_gguf(model_path, output_name):
    """Convert HuggingFace model to GGUF format"""
    f16_file = f"{output_name}-f16.gguf"

    # Check if conversion script exists
    convert_script = "./llama/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print(f"❌ Conversion script not found: {convert_script}")
        print("Make sure you're running this from the llama.cpp directory")
        return None

    command = (
        f"python3 {convert_script} ./{model_path} --outfile {f16_file} --outtype f16"
    )

    success = run_command(command, f"Converting {model_path} to GGUF format")

    if success and os.path.exists(f16_file):
        return f16_file
    else:
        print(f"❌ Failed to create {f16_file}")
        return None


def quantize_model(f16_file, output_name, quant_types):
    """Quantize the GGUF model with specified quantization types"""
    quantizer = "./llama/bin/llama-quantize"

    # Check if quantizer exists
    if not os.path.exists(quantizer):
        print(f"❌ Quantizer not found: {quantizer}")
        print("Make sure you have the llama.cpp release binary")
        return False

    successful_quants = []
    failed_quants = []

    for quant_type in quant_types:
        output_file = f"{output_name}-{quant_type}.gguf"
        command = f"{quantizer} {f16_file} {output_file} {quant_type}"

        success = run_command(command, f"Quantizing to {quant_type}")

        if success and os.path.exists(output_file):
            successful_quants.append(quant_type)
            # Get file size for reference
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"📦 Created {output_file} ({file_size:.1f} MB)")
        else:
            failed_quants.append(quant_type)

    return successful_quants, failed_quants


def main():
    """Main function"""
    try:
        # Get user input
        model_path, output_name, selected_quants = get_user_input()

        print("\nStarting conversion process...")
        print(f"Model path: {model_path}")
        print(f"Output name: {output_name}")
        print(f"Quantization types: {', '.join(selected_quants)}")

        # Step 1: Convert to GGUF
        print("\nStep 1: Converting to GGUF format")
        f16_file = convert_to_gguf(model_path, output_name)

        if not f16_file:
            print("❌ Conversion failed. Exiting.")
            sys.exit(1)

        # Get f16 file size
        f16_size = os.path.getsize(f16_file) / (1024 * 1024)  # MB
        print(f"📦 Created {f16_file} ({f16_size:.1f} MB)")

        # Step 2: Quantize
        print("\nStep 2: Quantizing model")
        successful_quants, failed_quants = quantize_model(
            f16_file, output_name, selected_quants
        )

        # Summary
        print("\nProcess completed!")
        print("=" * 50)

        if successful_quants:
            print(f"✅ Successfully created {len(successful_quants)} quantized models:")
            for quant in successful_quants:
                print(f"   - {output_name}-{quant}.gguf")

        if failed_quants:
            print(f"❌ Failed to create {len(failed_quants)} quantized models:")
            for quant in failed_quants:
                print(f"   - {output_name}-{quant}.gguf")

        # List all created files
        print("\n📁 Processing completed successfully!")
        print(f"All GGUF files have been moved to: {gguf_folder}/")

        # Move all GGUF files to separate folder
        print("\nMoving GGUF files to separate folder...")
        gguf_folder = f"{output_name}-GGUF"

        try:
            # Create the folder if it doesn't exist
            os.makedirs(gguf_folder, exist_ok=True)
            print(f"Created folder: {gguf_folder}/")

            moved_files = []

            # Move f16 base file
            if os.path.exists(f16_file):
                new_f16_path = os.path.join(gguf_folder, f16_file)
                os.rename(f16_file, new_f16_path)
                moved_files.append(f16_file)
                print(f"Moved {f16_file}")

            # Move all quantized files
            for quant in successful_quants:
                quant_file = f"{output_name}-{quant}.gguf"
                if os.path.exists(quant_file):
                    new_quant_path = os.path.join(gguf_folder, quant_file)
                    os.rename(quant_file, new_quant_path)
                    moved_files.append(quant_file)
                    print(f"Moved {quant_file}")

            print(
                f"\nSuccessfully moved {len(moved_files)} GGUF files to {gguf_folder}/"
            )

            # Show final folder contents with sizes
            print(f"\nFinal contents of {gguf_folder}/:")
            try:
                for filename in sorted(os.listdir(gguf_folder)):
                    if filename.endswith(".gguf"):
                        filepath = os.path.join(gguf_folder, filename)
                        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                        print(f"   {filename} ({size:.1f} MB)")
            except Exception as e:
                print(f"   Could not list folder contents: {e}")

        except Exception as e:
            print(f"Failed to create/move files to {gguf_folder}/: {e}")
            print("Files remain in current directory")

        # Optional cleanup of f16 file
        f16_in_folder = os.path.join(gguf_folder, f16_file)
        if os.path.exists(f16_in_folder) and successful_quants:
            cleanup = (
                input(
                    f"\nRemove the f16 base file ({f16_file}) from {gguf_folder}/? (y/n): "
                )
                .strip()
                .lower()
            )
            if cleanup == "y":
                try:
                    os.remove(f16_in_folder)
                    print(f"Removed {f16_file} from {gguf_folder}/")
                except Exception as e:
                    print(f"Failed to remove {f16_file}: {e}")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
