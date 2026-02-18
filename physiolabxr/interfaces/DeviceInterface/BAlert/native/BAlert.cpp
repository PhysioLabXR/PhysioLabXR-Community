// BAlert.cpp — ABM B-Alert RAW -> LSL (portable build + PhysioLabXR communication)
// Build: x64 / Release
// Link:  BAlert64.lib, ABM_ThirdPartyCommunication64.lib, lsl.lib

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cwchar>

#include "AbmSdkInclude.h"
#pragma comment(lib, "BAlert64.lib")
#pragma comment(lib, "ABM_ThirdPartyCommunication64.lib")

#include <lsl_cpp.h>
#pragma comment(lib, "lsl.lib")

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// ---------------- CONSTANTS ----------------
static const int  kDetectTimeoutSec = 20;
static const int  kRetries = 3;
static const int  kSampleRateHz = 256;
static const BOOL kAllowOldFW = TRUE;
static const BOOL kAllowNoStrip = FALSE; 

static const char* kStatusFile = "balert_status.txt";
static const char* kCommandFile = "balert_command.txt";

// ---------------- GLOBALS ----------------
static std::wstring g_cfgDir, g_binDir, g_licDir; 
static std::atomic<bool> g_doneImpedance{ false };
static std::atomic<bool> g_stop{ false };
static std::atomic<bool> g_ready{ false };

// ---------------- PhysioLabXR ----------------
enum Status {
    STATUS_STARTING = 0,
    STATUS_INITIALIZING,
    STATUS_DETECTING,
    STATUS_MEASURING_IMPEDANCE,
    STATUS_READY,
    STATUS_STREAMING,
    STATUS_ERROR,
    STATUS_STOPPING,
    STATUS_STOPPED
};

static void write_status(Status st, const std::string& msg = std::string()) {
    std::ofstream f(kStatusFile, std::ios::trunc);
    if (!f.good()) return;
    f << static_cast<int>(st);
    if (!msg.empty()) f << "," << msg;
    f.close();
}

static bool take_stop_command() {
    std::ifstream f(kCommandFile);
    if (!f.good()) return false;
    std::string cmd;
    std::getline(f, cmd);
    f.close();
    if (cmd == "STOP" || cmd == "stop") {
        std::ofstream c(kCommandFile, std::ios::trunc);
        c.close();
        return true;
    }
    return false;
}

// ---------------- small helpers ----------------
static BOOL WINAPI ConsoleHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT ||
        type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT ||
        type == CTRL_SHUTDOWN_EVENT) {
        g_stop = true;
        write_status(STATUS_STOPPING, "Console interrupt");
        return TRUE;
    }
    return FALSE;
}

static std::wstring exe_dir() {
    wchar_t buf[MAX_PATH]{ 0 };
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring p(buf);
    size_t pos = p.find_last_of(L"\\/");
    return (pos == std::wstring::npos) ? L"." : p.substr(0, pos);
}

static std::wstring join2(const std::wstring& a, const std::wstring& b) {
    if (a.empty()) return b;
    return (a.back() == L'\\' || a.back() == L'/') ? (a + b) : (a + L"\\" + b);
}

static void set_env_w(const wchar_t* name, const std::wstring& val) {
    SetEnvironmentVariableW(name, val.c_str());
}

static bool detect_paths() {
    write_status(STATUS_STARTING, "Detecting paths");

    const std::wstring ed = exe_dir();

    std::vector<std::wstring> bases = {
        join2(ed, L"third_party\\BAlert"),
        join2(ed, L"third_party\\balert"),
        join2(join2(ed, L".."), L"third_party\\BAlert"),
        join2(join2(ed, L".."), L"third_party\\balert"),
        join2(join2(join2(ed, L".."), L".."), L"third_party\\BAlert"),
        join2(join2(join2(ed, L".."), L".."), L"third_party\\balert"),
        ed // fallback
    };

    for (auto& base : bases) {
        const std::wstring cfg = join2(base, L"Config\\");
        const std::wstring bin = join2(base, L"SDK64\\bin\\");
        const std::wstring lic = join2(base, L"License\\");
        std::wcout << L"[BAlert] Try base: " << base << L"\n";

        if (fs::exists(cfg) && fs::exists(bin) && fs::exists(join2(cfg, L"AthenaSDK.xml"))) {
            g_cfgDir = cfg; g_binDir = bin; g_licDir = lic;
            std::wcout << L"[BAlert] Use bundled ABM: cfg=" << g_cfgDir
                << L" bin=" << g_binDir << L" lic=" << g_licDir << L"\n";
            return true;
        }
    }

    const std::wstring inst = L"C:\\ABM\\B-Alert";
    const std::wstring cfg = join2(inst, L"Config\\");
    const std::wstring bin = join2(inst, L"SDK64\\bin\\");
    const std::wstring lic = join2(inst, L"License\\");
    if (fs::exists(cfg) && fs::exists(bin) && fs::exists(join2(cfg, L"AthenaSDK.xml"))) {
        g_cfgDir = cfg; g_binDir = bin; g_licDir = lic;
        std::wcout << L"[BAlert] Use installed ABM: cfg=" << g_cfgDir
            << L" bin=" << g_binDir << L" lic=" << g_licDir << L"\n";
        return true;
    }
    return false;
}

static void apply_runtime_or_throw() {
    if (!detect_paths()) {
        write_status(STATUS_ERROR, "B-Alert files not found");
        throw std::runtime_error("B-Alert installation not found");
    }

    SetDllDirectoryW(g_binDir.c_str());

    set_env_w(L"ABM_LICENSE_DIR", g_licDir);

    if (!SetCurrentDirectoryW(g_cfgDir.c_str())) {
        write_status(STATUS_ERROR, "SetCurrentDirectory failed");
        throw std::runtime_error("SetCurrentDirectory failed");
    }

    if (SetConfigPath(const_cast<wchar_t*>(g_cfgDir.c_str())) == 0) {
        write_status(STATUS_ERROR, "SetConfigPath failed");
        throw std::runtime_error("SetConfigPath failed");
    }

    std::wcout << L"[BAlert] exe_dir=" << exe_dir() << L"\n";
    std::wcout << L"[BAlert] cfg_dir=" << g_cfgDir << L"\n";
    std::wcout << L"[BAlert] bin_dir=" << g_binDir << L"\n";
    std::wcout << L"[BAlert] lic_dir=" << g_licDir << L"\n";
}

// ---------------- ABM helpers ----------------
static void CallbackImpedance(ELECTRODE* pEl, int& nCount) {
    for (int i = 0; i < nCount; ++i)
        std::wcout << L"[BAlert] Impedance " << pEl[i].chName
        << L" = " << pEl[i].fImpedanceValue << L"\n";
    g_doneImpedance = true;
}

static std::wstring devtype_to_string(int id) {
    switch (id) {
    case ABM_DEVICE_X24Flex_10_20:        return L"X24Flex_10_20";
    case ABM_DEVICE_X24Flex_10_20_LM:     return L"X24Flex_10_20_LM";
    case ABM_DEVICE_X24LE_10_20_LM:       return L"X24LE_10_20_LM";
    case ABM_DEVICE_X24Flex_10_20_LM_Red: return L"X24Flex_10_20_LM_Red";
    case ABM_DEVICE_X24LE_10_20_LM_Red:   return L"X24LE_10_20_LM_Red";
    case ABM_DEVICE_X10Flex_Standard:     return L"X10Flex_Standard";
    case ABM_DEVICE_X24Flex_Reduced:      return L"X24Flex_Reduced";
    case ABM_DEVICE_X10Flex_10_20_LM_Red: return L"X10Flex_10_20_LM_Red";
    default: return L"Unknown(" + std::to_wstring(id) + L")";
    }
}

static void print_connection_state() {
    BOOL opened = IsConnectionOpened();
    auto* esu = GetESUPortInfo();
    std::wcout << L"[ABM] IsConnectionOpened=" << (opened ? L"TRUE" : L"FALSE") << L"\n";
    if (esu)
        std::wcout << L"[ABM] ESU: type=" << esu->nESU_TYPE
        << L" wired=" << (esu->bWired ? L"TRUE" : L"FALSE") << L"\n";
}

static _DEVICE_INFO* detect_device() {
    write_status(STATUS_DETECTING, "Detecting device");

    CloseCurrentConnection();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    for (int a = 1; a <= kRetries; ++a) {
        if (g_stop.load() || take_stop_command()) {
            write_status(STATUS_STOPPING, "Stop during detect");
            return nullptr;
        }
        int strip = -1, err = -1;
        _DEVICE_INFO* dev = GetDeviceInfo(kDetectTimeoutSec, strip, err);
        if (dev) {
            std::wcout << L"[BAlert] Device: " << dev->chDeviceName
                << L", channels=" << dev->nNumberOfChannel
                << L", handle=" << dev->nDeviceHandle << L"\n";
            return dev;
        }
        std::wcout << L"[BAlert] Detect attempt " << a << L"/" << kRetries
            << L" failed (strip=" << strip << L", err=" << err << L")\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    write_status(STATUS_ERROR, "No device detected");
    return nullptr;
}

static int try_init_raw(_DEVICE_INFO* dev) {
    if (!dev) return -1;

    const bool is24 = (dev->nNumberOfChannel >= 24);
    std::vector<int> ids = is24
        ? std::vector<int>{6, 9, 10, 11, 12}
    : std::vector<int>{ 7,8,13 };

    const int handles[] = { 0, -1 };
    for (int id : ids) {
        for (int h : handles) {
            if (g_stop.load() || take_stop_command()) {
                write_status(STATUS_STOPPING, "Stop during init");
                return -1;
            }
            int rc = InitSessionForCurrentConnection(id, ABM_SESSION_RAW, h, FALSE);
            std::wcout << L"[BAlert] Init RAW: id=" << id << L" ("
                << devtype_to_string(id) << L") handle=" << h
                << L" rc=" << rc << L"\n";
            if (rc == 1) return id;

            rc = InitSessionForCurrentConnection(id, ABM_SESSION_RAW, h, TRUE);
            std::wcout << L"[BAlert] Init RAW(EBS=TRUE): id=" << id << L" ("
                << devtype_to_string(id) << L") handle=" << h
                << L" rc=" << rc << L"\n";
            if (rc == 1) return id;
        }
    }
    return -1;
}

static bool measure_impedances() {
    write_status(STATUS_MEASURING_IMPEDANCE, "Measuring impedances");

    if (MeasureImpedances(CallbackImpedance) != IMP_STARTED_OK) {
        write_status(STATUS_ERROR, "MeasureImpedances failed to start");
        return false;
    }
    while (!g_doneImpedance.load()) {
        if (g_stop.load() || take_stop_command()) {
            write_status(STATUS_STOPPING, "Stop during impedance");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return true;
}

static std::unique_ptr<lsl::stream_outlet> make_outlet(int nRawCh, int device_id) {
    _CHANNELMAP_INFO cmap{}; bool haveMap = (GetChannelMapInfo(cmap) != 0);

    lsl::stream_info info("BAlert", "EEG", nRawCh, kSampleRateHz,
        lsl::cf_float32, "BAlert_RAW_" + std::to_string(device_id));

    auto acq = info.desc().append_child("acquisition");
    acq.append_child_value("manufacturer", "Advanced Brain Monitoring");
    acq.append_child_value("model", "B-Alert X-Series");

    // device_type: wide->narrow
    std::wstring w = devtype_to_string(device_id);
    size_t n = w.size() + 1;
    std::vector<char> tmp(n, 0);
    size_t outc = 0;
    wcstombs_s(&outc, tmp.data(), n, w.c_str(), n - 1);
    acq.append_child_value("device_type", tmp.data());

    auto chs = info.desc().append_child("channels");
    for (int i = 0; i < nRawCh; ++i) {
        auto c = chs.append_child("channel");
        if (haveMap) c.append_child_value("label", cmap.stEEGChannels.cChName[i]);
        else {
            std::string lab = "EEG" + std::to_string(i + 1);
            c.append_child_value("label", lab.c_str());
        }
        c.append_child_value("unit", "microvolts");
        c.append_child_value("type", "EEG");
    }

    std::wcout << L"[BAlert] LSL outlet ready: ch=" << nRawCh
        << L" srate=" << kSampleRateHz << L"\n";

    return std::make_unique<lsl::stream_outlet>(info);
}

static void stream_loop(lsl::stream_outlet& outlet, int nRawCh) {
    write_status(STATUS_READY, "Ready to stream");
    g_ready = true;

    std::wcout << L"[BAlert] StartAcquisition...\n";
    int rc = StartAcquisition();
    if (rc != 1) {
        write_status(STATUS_ERROR, "StartAcquisition failed");
        throw std::runtime_error("StartAcquisition failed");
    }

    write_status(STATUS_STREAMING, "Streaming");
    std::wcout << L"[BAlert] Streaming... (write STOP in balert_command.txt to stop)\n";

    auto t0 = std::chrono::steady_clock::now();
    long long pushed = 0;

    while (!g_stop.load() && !take_stop_command()) {
        int nRecv = 0;
        float* buf = GetRawData(nRecv);
        if (buf && nRecv > 0) {
            for (int i = 0; i < nRecv; ++i)
                outlet.push_sample(&buf[i * nRawCh]);
            pushed += nRecv;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (pushed && (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5))) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::wcout << L"[BAlert] " << pushed << L" samples, ~"
                << std::fixed << std::setprecision(1) << (pushed / dt) << L" Hz\n";
            t0 = std::chrono::steady_clock::now();
            pushed = 0;
        }
    }

    write_status(STATUS_STOPPING, "Stopping");
    std::wcout << L"[BAlert] Stopping...\n";
    StopAcquisition();
    CloseCurrentConnection();
}

// ---------------- MAIN ----------------
int wmain() {
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    try {
        apply_runtime_or_throw();

        SetClassicFlexOldFwAllowed(kAllowOldFW);
        SetNoStripAllowed(kAllowNoStrip);

        write_status(STATUS_INITIALIZING, "Initializing SDK");
        print_connection_state();

        _DEVICE_INFO* dev = detect_device();
        if (!dev) return 1;

        int device_id = try_init_raw(dev);
        if (device_id < 0) {
            write_status(STATUS_ERROR, "Init RAW failed");
            return 1;
        }

        if (!measure_impedances()) return 1;

        device_id = try_init_raw(dev);
        if (device_id < 0) {
            write_status(STATUS_ERROR, "Re-init RAW failed");
            return 1;
        }

        int nRaw = 0, nDecon = 0, nPSD = 0, nRawPSD = 0, nQual = 0;
        GetPacketChannelNmbInfo(nRaw, nDecon, nPSD, nRawPSD, nQual);
        if (nRaw <= 0) nRaw = dev->nNumberOfChannel;
        std::wcout << L"[BAlert] Channels: RAW=" << nRaw
            << L" Decon=" << nDecon << L" PSD=" << nPSD << L"\n";

        auto outlet = make_outlet(nRaw, device_id);
        stream_loop(*outlet, nRaw);
    }
    catch (const std::exception& e) {
        std::wcerr << L"[BAlert] EXCEPTION: " << e.what() << L"\n";
        write_status(STATUS_ERROR, e.what());
        try { StopAcquisition(); CloseCurrentConnection(); }
        catch (...) {}
        return 1;
    }
    catch (...) {
        std::wcerr << L"[BAlert] UNKNOWN EXCEPTION\n";
        write_status(STATUS_ERROR, "Unknown exception");
        try { StopAcquisition(); CloseCurrentConnection(); }
        catch (...) {}
        return 1;
    }

    write_status(STATUS_STOPPED, "Stopped OK");
    std::wcout << L"[BAlert] Done.\n";
    return 0;
}