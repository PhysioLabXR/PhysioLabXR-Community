# Building BAlert Executables

This guide provides instructions on building the `BAlert.exe`executables in Visual Studio.

## Prerequisites
- **Visual Studio**: Installed with the C++ Desktop development workload.
- **LSL(liblsl)**: Add `include`/`lib`, link `lsl.lib`, and put `lsl.dll` next to the exe in `x64/Release` (or add `bin` to `PATH`).
- **ABM SDK64**: Add headers/libs, link `BAlert64.lib` and `ABM_ThirdPartyCommunication64.lib`, and place required ABM DLLs by the exe.
- **ABM License folder** : Place the provided `License` folder under `x64/Release`.

## Build the Executables

1. **Open Solution in Visual Studio**
   - Open `BAlert.sln` in Visual Studio.

2. **Set the Startup Project**
   - In Solution Explorer, right-click on `BAlert` and select **Set as Startup Project**.

3. **Build BAlert.exe**
   - Go to the top menu and select **Build → Build Solution** or press **Ctrl+Shift+B**.

4. **Locate the Executables**
   - After successful builds, BAlert.exe will be located in `x64/Release` folder.

## Troubleshooting
- Ensure that all dependencies are properly linked.
- Ensure that License folder is cloned properly.
- Verify that the correct build configuration (`Release`) is selected.

You’re all set! The executables are now ready to use.