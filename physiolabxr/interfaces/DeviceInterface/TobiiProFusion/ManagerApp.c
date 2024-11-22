#include <windows.h>
#include <stdio.h>
#include "Shlwapi.h"
#include "tobii_research.h"
#include "tobii_research_eyetracker.h"

//static void GetEyeTrackerManagerPath(wchar_t* path, DWORD size) {
//    if (!GetEnvironmentVariableW(L"LocalAppData", path, size)) {
//        fwprintf(stderr, L"Local App not found.\n");
//        exit(EXIT_FAILURE);
//    }
//    PathAppendW(path, L"Programs\\TobiiProEyeTrackerManager\\TobiiProEyeTrackerManager.exe");
//    fwprintf(stderr, L"ETM Path set to: %ls\n", path); // Debug statement
//}

static void AddArgsToEyeTrackerManagerCall(char* call, DWORD call_size, const char* deviceaddress) {
    _snprintf_s(call, call_size, _TRUNCATE, "\"%s\" --device-address=%s --mode=displayarea", call, deviceaddress);
    printf("ETM Call Command: %s\n", call); // Debug statement
}

static void PrintfLinesStartingWith(HANDLE f, const char* line_start) {
    DWORD bytes_read;
#define BUF_SIZE 64
    CHAR buf[BUF_SIZE];
    BOOL success = FALSE;
    BOOL found_matching_line_start = FALSE;
    size_t line_start_len = strlen(line_start);
    size_t bytes_to_read = line_start_len;

    if (line_start_len > BUF_SIZE) {
        printf("ERROR: %s longer than BUF_SIZE!\n", line_start);
        exit(EXIT_FAILURE);
    }

    do {
        ZeroMemory(buf, sizeof(buf));
        success = ReadFile(f, buf, (DWORD)bytes_to_read, &bytes_read, NULL);

        if (bytes_read == 1 && buf[0] == '\n') {
            found_matching_line_start = FALSE;
            bytes_to_read = line_start_len;
        }
        else if (bytes_read == line_start_len && !strncmp(buf, line_start, line_start_len)) {
            found_matching_line_start = TRUE;
            bytes_to_read = 1;
        }

        if (found_matching_line_start) {
            printf("%s", buf);
        }

    } while (success || bytes_read != 0);
}

static void GetEyeTrackerAddresses(char* first_address, DWORD size) {
    TobiiResearchEyeTrackers* eyetrackers = NULL;
    TobiiResearchStatus result = tobii_research_find_all_eyetrackers(&eyetrackers);

    if (result != TOBII_RESEARCH_STATUS_OK) {
        printf("Finding trackers failed. Error: %d\n", result);
        exit(EXIT_FAILURE);
    }

    printf("Found %d Eye Trackers\n", (int)eyetrackers->count); // Debug statement

    for (size_t i = 0; i < eyetrackers->count; i++) {
        TobiiResearchEyeTracker* eyetracker = eyetrackers->eyetrackers[i];
        char* address = NULL;
        char* serial_number = NULL;
        char* device_name = NULL;

        tobii_research_get_address(eyetracker, &address);
        tobii_research_get_serial_number(eyetracker, &serial_number);
        tobii_research_get_device_name(eyetracker, &device_name);

        printf("Address: %s, Serial Number: %s, Device Name: %s\n", address, serial_number, device_name);

        if (i == 0) { // Save the first address to use in the ETM call
            strncpy_s(first_address, size, address, _TRUNCATE);
        }

        tobii_research_free_string(address);
        tobii_research_free_string(serial_number);
        tobii_research_free_string(device_name);
    }

    tobii_research_free_eyetrackers(eyetrackers);
}

#pragma comment(lib, "Shlwapi.lib")
int wmain(int argc, wchar_t* argv[]) {
    if (argc < 2) {
        fwprintf(stderr, L"Usage: %s <path_to_eye_tracker_manager_exe>\n", argv[0]);
        return 1;
    }
    wchar_t* ETM_CALL = argv[1];
    //wchar_t ETM_CALL[MAX_PATH * 2] = { 0 };  // Ensure enough space
    char first_address[MAX_PATH] = { 0 };
    /*GetEyeTrackerManagerPath(ETM_CALL, sizeof(ETM_CALL) / sizeof(wchar_t));*/
    GetEyeTrackerAddresses(first_address, MAX_PATH);

    // Simulated function for getting the address
    /*strncpy_s(first_address, MAX_PATH, "tobii-prp://TPFC2-010203826601", _TRUNCATE);*/

    char command_line[MAX_PATH * 2] = { 0 };
    AddArgsToEyeTrackerManagerCall(command_line, sizeof(command_line), first_address);

    STARTUPINFOW siStartInfo = { sizeof(siStartInfo) };
    PROCESS_INFORMATION piProcInfo;

    if (!CreateProcessW(ETM_CALL, command_line, NULL, NULL, TRUE, 0, NULL, NULL, &siStartInfo, &piProcInfo)) {
        fwprintf(stderr, L"Create Process failed with error %lu\n", GetLastError());
        return 1;
    }

    WaitForSingleObject(piProcInfo.hProcess, INFINITE);
    CloseHandle(piProcInfo.hProcess);
    CloseHandle(piProcInfo.hThread);

    return 0;
}
