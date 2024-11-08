# Building TobiiProFusion Executables

This guide provides instructions on building the `TobiiProFusion_Process.exe` and `CallManagerApp.exe` executables in Visual Studio.

## Prerequisites
- **Visual Studio**: Make sure you have Visual Studio installed with support for C++ development.
- **Tobii SDK**: Ensure all necessary Tobii SDK libraries and dependencies are included in the solution.

## Build the Executables

1. **Open Solution in Visual Studio**
   - Open `TobiiProFusion_Process.sln` in Visual Studio.

2. **Set the Startup Project**
   - In Solution Explorer, right-click on `TobiiProFusion_Process` and select **Set as Startup Project**.

3. **Build TobiiProFusion_Process.exe**
   - Go to the top menu and select **Build → Build Solution** or press **Ctrl+Shift+B**.

4. **Repeat for CallManagerApp**
   - Right-click on `CallManagerApp` in Solution Explorer, select **Set as Startup Project**, and build the solution again.

5. **Locate the Executables**
   - After successful builds, the executables (`TobiiProFusion_Process.exe` and `CallManagerApp.exe`) will be located in either `x64/Debug` or `x64/Release` folders, depending on your build configuration.

## Troubleshooting
- Ensure that all dependencies are properly linked.
- Verify that the correct build configuration (`Debug` or `Release`) is selected.

You’re all set! The executables are now ready to use.
