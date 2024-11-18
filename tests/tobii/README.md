# Testing TobiiProFusion_Options UI

This README provides instructions for manually testing the `TobiiProFusion_Options` UI component using a `pytest` setup. The test will load the UI, allowing you to manually interact with it.

## Prerequisites

Ensure the following are installed:
- `pytest`
- `pytest-qt`
- `PyQt6`
- `unittest.mock`

## Usage

1. **Set Up**:
   - Clone or download the repository and navigate to the test folder where this file is located.
   - Ensure that the path to `TobiiProFusion_Options.ui` in the test file is correct for your system. The current path is set as follows:
     ```python
     setattr(AppConfigs(), '_ui_TobiiProFusion_Options', 
             r'C:\Users\Zeyi Tong\Desktop\PhysioLabXR\PhysioLabXR-Community\physiolabxr\_ui\TobiiProFusion_Options.ui')
     ```

2. **Run the Test**:
   - Execute the test using the following command:
     ```bash
     pytest <test_filename>.py
     ```
     Replace `<test_filename>` with the name of the test file (e.g., `test_tobiipro_options.py`).

3. **Manual Testing Instructions**:
   - Once the test begins, the `TobiiProFusion_Options` UI will open for manual interaction.
   - You will have `8000 ms` (or 8 seconds) to interact with the UI. Modify this delay as needed:
     ```python
     qtbot.wait(8000)  # Adjust as needed for testing
     ```
   - **Click the Button**:
     - Locate the button you want to test within the UI.
     - Click the button and observe the response. This could include dialog boxes, text updates, or any other UI behavior triggered by the button.
   - Verify that the button works as expected:
     - Confirm if clicking the button triggers the desired functionality or displays the correct output.

4. **Expected Output**:
   - The UI should display, and "UI is open for manual testing" will print to the console.
   - Any output or behavior associated with the button click should be observed and verified.
