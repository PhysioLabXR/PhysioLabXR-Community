#ifdef BMS_EXPORTS
#define BMS_API __declspec(dllexport)
#else
#define BMS_API __declspec(dllimport)
#endif

#include "typedef.h"
#include ".\BMS_Comm\BMS_Comm.h"
#include "SDKFlexDefinitions.h"

//         Desc: This function returns information on Balert device, it is optional and may be called at any time. 
//               It is advisable to call this function at the beginning of a session to ensure that the device is active and connected successfully. 
//   Parameters: &deviceInfo - OUTPUT - _BALERT_DEVICE_INFO_BMS struct holding information on device
//			     nDeviceType - INPUT - a type of Balert device (X4-BAlert = 4, X10-Standard=10, X24-qEEG=24, X24-stERP=241)
// Return Value: Connection status (1 = Device & ESU connected and working fine , 0=no device, -1= wrong device model selected, -2 = ESU or device not configured properly)
BMS_API int __stdcall BAlertGetDeviceInfo(_BALERT_DEVICE_INFO_BMS &deviceInfo, int nDeviceType); 



//         Desc: This function sets destination file, if not called prior to start of the acquisition, then no output file will be created
//               Current date and time is being appended to the filename and two files will be created (Signals.EDF and Events.EDF) for every session 
//               This function may be called multiple times while data acquisition and impedance checking are not active and each time 
//               a new set of EDF+ files will be created with a unique name appended with the current date and time. 
//   Parameters: pDestinationFilePath - INPUT - full path of destination file (e.g. C:\\ABM\\EEG\\Data)
//				 nSubjNum - INPUT - a session information parameter
//	             nGroup  - INPUT - a session information parameter
//               nSessIter - INPUT - a session information parameter
//               nTaskType - INPUT - a session information parameter
//               nTaskIter - INPUT - a session information parameter
//				 nDevice type: INPUT -  a type of Balert device (X4-BAlert = 4, X10-Standard=10, X24-qEEG=24, X24-stERP=241) 
// Return Value: status - Success (1)/ Failed (0) 
BMS_API int __stdcall BAlertSetDestinationFile( WCHAR* pDestinationFilePath, int nSubjNum, int nGroup, int nSessIter, int nTaskType, int nTaskIter, int nDeviceType, bool bSecurityMode );

//         Desc: This function starts impedance measurement - it opens the communication port if the port is not already opened. 
//   Parameters: pFunc - INPUT - a callback function - reports each channel separately - the impedance associated with both the electrodes of that channel are returned.
//				 channels - INPUT - a channel selection mask - integer value (4 bytes) coded as bits - each bit for one channel (0xFFFFFFFF is reserved to indicate all channels)
//									(for e.g to check impedance of channel 2, the user should enter 0x00000002, to check channels 2 & 3 enter 0x00000006)
//			     nDeviceType - INPUT - a type of Balert device (X4-BAlert = 4, X10-Standard=10, X24-qEEG=24, X24-stERP=241)
// Return Value: status (success=1/failed=0)
BMS_API int __stdcall BAlertCheckImpedances(void (__stdcall *pFunc)(_BALERT_IMPEDANCE_RESULT*, int&), int channels, int nDeviceType);

//         Desc: This function starts data Acquisition - it opens the communication port if the port is not already opened.
//   Parameters: None
// Return Value:  status  - (success (1) /failed (0)). 
BMS_API int __stdcall BAlertStartAcquisition();

//         Desc: This function ends the session and closes all active files. 
//   Parameters: None
// Return Value: status - (success (1) /failed (0)). 
BMS_API int __stdcall BAlertStopAcquisition();


//         Desc:  Stops impedance measurement in progress.
//   Parameters:  none
// Return Value:  (success=1/failed=0). 
BMS_API int __stdcall BAlertStopImpedance(BOOL bCanceled);

//         Desc: This function is blocking, it blocks till a number of samples equal to nBlockSize is available from each EEG channel. 
//               It flushes the internal buffer the very first time this function is called and resets the samplecounter 
//   Parameters: fbuffer - OUTPUT - a pointer to buffer holding data
//               The return buffer has structure which depends on device used. It can be one of the folowing:
//               _BALERT_DATA_X10_PACKET, _BALERT_DATA_X10_PACKET or _BALERT_DATA_X24_PACKET.
//               The total size of the return buffer will be fixed and equal to nBlockSize * sizeof (_BALERT_DATA_XYZ_PACKET) 
//               nBlockSize - INPUT - a number of samples to wait for
// Return Value: status (success-1/failed-0)
BMS_API int __stdcall BAlertWaitForData(float* fBuffer, int nBlockSize);

//         Desc: This function is NON-blocking and returns all the data available in the internal buffer at that moment of call. 
//               The return buffer has structure which depends on device used. It can be one of the folowing:
//               _BALERT_DATA_X10_PACKET, _BALERT_DATA_X10_PACKET or _BALERT_DATA_X24_PACKET.
//   Parameters: &nCount - OUTPUT -  number of samples returned per channel.
// Return Value: pointer to the buffer holding data - The size of the return buffer is equal to nCount * sizeof (_BALERT_DATA_XYZ_PACKET)
BMS_API float* __stdcall BAlertGetData(int &nCount);


//         Desc: This function is NON-blocking and returns all the MIC data available in the internal buffer at that moment of call. 
//				 (works only for X4 devices with extended protocol)
//   Parameters: &nCount - OUTPUT -  number of samples returned per channel.
// Return Value: pointer to the buffer holding data 
BMS_API float* __stdcall BAlertGetMicData(int &nCount);

//         Desc: This function is NON-blocking and returns all the Raw Optical data available in the internal buffer at that moment of call. 
//				 (works only for X4 devices with extended protocol)
//   Parameters: &nCount - OUTPUT -  number of samples returned per channel.
// Return Value: pointer to the buffer holding data 
BMS_API float* __stdcall BAlertGetRawOpticalData(int &nCount);

//         Desc: This function is NON-blocking and returns all the Optical data available in the internal buffer at that moment of call. 
//				 (works only for X4 devices with extended protocol)
//   Parameters: &nCount - OUTPUT -  number of samples returned per channel.
// Return Value: pointer to the buffer holding data 
BMS_API float* __stdcall BAlertGetOpticalData(int &nCount);

//		   Desc: The function is non-blocking and returns all the available events in the event buffer. 
//   Parameters: nCount - OUTPUT - number of event packets in the event buffer. 
// Return Value: Event Buffer - has a structure like in _BALERT_DATA_EVENT and can be explicitly casted to a pointer to it
//	 			 Size of the buffer is nCount*sizeof ( _BALERT_DATA_EVENT)
BMS_API BYTE* __stdcall BAlertGetEvents(int &nCount);


//         Desc:  This function enables calling applications to send events to the dll which will then be timestamped, 
//				  stored in the internal buffer and in EDF file (if available), and made accessible through BAlertGetEvents function. 
//                Events MUST be US-ASCII encoded, non-ASCII characters out of the limits will be dropped and will not be stored.
//   Parameters:  pBuffer - INPUT -  Char buffer pointer holding text
//				  nEventSize - INPUT - buffer size in bytes
// Return Value:  Status - (1=event timestamped and stored, 0=failed).
BMS_API int __stdcall BAlertSetEvents(WCHAR* pBuffer, int nEventSize, WORD nHour, WORD nMinute, WORD nSecond, WORD nMiliSec );

//		  Desc: Get firmware version from either headset or dongle or esu device
//	Parameters: - INPUT - nType - headset=1, dongle=2, esu=3
//				- OUTPUT - pcVersion - char[50], return version as xx.xx.xx
// Retrun Value: on success return 1, if failed return -1
BMS_API int __stdcall BAlertGetFirmwareVer(int nType, WCHAR* pcVersion);


//		  Desc: Disconnect device
//	Parameters: 
//	Return Value: on success return 1, if failed return -1
BMS_API int __stdcall BAlertCloseSession();

BMS_API int __stdcall BAlertSetCallbackOnNewSample(DWORD dwDAUHandle, PABMOnNewSample pOnNewSample, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewRawSample(DWORD dwDAUHandle, PABMOnNewRawSample pOnNewRawSample, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewTimestamp(DWORD dwDAUHandle, PABMOnNewTimestamp pOnNewSample, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewTilt(DWORD dwDAUHandle, PABMOnNewTilt pOnNewTilt, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewRawSample(DWORD dwDAUHandle, PABMOnNewRawSample pOnNewSample, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewThirdPartyData(DWORD dwDAUHandle, PABMOnNewThirdPartyData pOnNewThirdPartyData, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewSlowChannels(DWORD dwDAUHandle, PABMOnNewSlowChannels pOnNewSlowChannels, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewIRed(DWORD dwDAUHandle, PABMOnNewIRed pOnNewIRed, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewProbeData(DWORD dwDAUHandle, PABMOnNewProbeData pOnNewProbeData, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewMic(DWORD dwDAUHandle, PABMOnNewMicData pOnNewMic, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewIRedRaw(DWORD dwDAUHandle, PABMOnNewIRedRawData pOnNewIRedRaw, void* p);
BMS_API int __stdcall BAlertSetCallbackOnNewTechnicalChannel(DWORD dwDAUHandle, PABMOnNewTechnicalChannel pOnNewTechnicalChannel, void* p);


BMS_API int __stdcall BAlertGetDeviceSerialNum(WCHAR** serialNum);
BMS_API int __stdcall BAlertGetDAUInfo(TABMDAUInfo* info);
BMS_API int __stdcall BAlertGetESUInfo(TABMESUInfo* info);

BMS_API int __stdcall   BAlertGetRawSample(DWORD dwDAUHandle, TABMRawSampleInfo* pRawSampleInfo, DWORD dwCount, DWORD* dwActualCount);
BMS_API int __stdcall   BAlertGetSample2(DWORD dwDAUHandle, TABMSampleInfo* pRawSampleInfo, DWORD dwCount, DWORD* dwActualCount);
BMS_API float* __stdcall   BAlertGetSample(DWORD dwDAUHandle,  DWORD dwCount, DWORD* dwActualCount);

BMS_API bool __stdcall  BAlertImpedanceMeasurementStart(DWORD dwDAUHandle);
BMS_API bool __stdcall  BAlertImpedanceMeasurementStop(DWORD dwDAUHandle);
BMS_API bool __stdcall  BAlertImpedanceMeasurementSetChannel(DWORD dwDAUHandle, bool bHighCurrent, int channelIndex);
BMS_API bool __stdcall  BAlertSendBytesToDAU(DWORD dwDAUHandle,unsigned char* pBytes, int nBytes, int nRetry, int nPause);

BMS_API int __stdcall BAlertDisconnect();
BMS_API int __stdcall BAlertDisconnectLight(int);
BMS_API int __stdcall BAlertConnectLight();

BMS_API void __stdcall BAlertSetProcessMissedBlock(BOOL flag);

BMS_API void __stdcall BAlertSetWriteOutputs(BOOL flag);

BMS_API void __stdcall BAlertWriteFlexImpedanceResults( char* pchElectrodes, char* pchImpResults, int nCountElectrodes, int* nRawImpValues );


BMS_API int __stdcall BAlertStop();
BMS_API int __stdcall BAlertStopLight();
BMS_API void __stdcall BAlertGetEDFFileNames( WCHAR** pchSignalFileName, WCHAR** pchEventsFileName);

//used to set init battery state
BMS_API int __stdcall BAlertGetBatteryPercentage(float fBattery);


//used to set init fill device info usually for battery state
BMS_API int __stdcall BAlertFillDeviceInfo(_BALERT_DEVICE_INFO_BMS &deviceInfo);

BMS_API void __stdcall BAlertFreeBuffer( void* ptrBuffer);

//flex related api
BMS_API int __stdcall BAlertFlexGetSensorConfigurationBase(int&);

BMS_API int __stdcall BAlertFlexMeasureImpedances( int nSensorConfig, int** naImpedances );

BMS_API int __stdcall BAlertFlexUSBSynch(RECEIVER_INFO& esu, DEVICE_INFO& dev);

BMS_API int __stdcall BAlertGetFlexDeviceInfoViaUSB( FLEX_DEVICE_INFO& XDev );

BMS_API int __stdcall BAlertFlexUploadFirmware(WCHAR* fwFile);

BMS_API int __stdcall BAlertGetFlexDeviceModel(int& nDeviceModel, int&, float&);

BMS_API int __stdcall BAlertGetFlexBatteryEstimate();

//BMS_API int __stdcall BAlertGetFlexBatteryEstimateForCurrentConnection();

//New ST receiver(dongle) firmware upload
BMS_API int __stdcall BAlertUpgradeFirmwareReceiverST(WCHAR* fwFile, int );
BMS_API int __stdcall BAlertUpgradeFirmwareReceiverLE(WCHAR* fwFile);

BMS_API DWORD __stdcall BAlertEnumDevices(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p);
BMS_API DWORD __stdcall BAlertConnect();
BMS_API DWORD __stdcall BAlertDisconnectConnectionHandle(DWORD dwHandle);
BMS_API DWORD __stdcall BAlertEnumConnections(PABMEnumConnectionsProc pEnumConnectionsProc, void* p);
BMS_API DWORD __stdcall BAlertGetESUInfoForConnection(DWORD dwConnectionHandle, bool bReadFromMemory, TABMESUInfo* pESUInfo);
BMS_API DWORD __stdcall BAlertESUEditSettingsEnd(DWORD dwESUHandle, bool bSaveChanges);
BMS_API DWORD __stdcall BAlertESUSetBTPortSettings(DWORD dwESUHandle, TABMESUBTPortSettings* pBTPortSettings, BYTE bPortIndex);
BMS_API DWORD __stdcall BAlertESUSetPARPortSettings(DWORD dwESUHandle, TABMESUPARPortSettings* pPARPortSettings);
BMS_API DWORD __stdcall BAlertESUEditSettingsBegin(DWORD dwESUHandle);
BMS_API DWORD __stdcall BAlertESUSetSERPortSettings(DWORD dwESUHandle, TABMESUSERPortSettings* pSERPortSettings, BYTE bPortIndex);
BMS_API DWORD __stdcall BAlertEnumPorts(PABMEnumPortsProc pEnumPortsProc, TABMPortInfoEx* p);
BMS_API DWORD __stdcall BAlertBTSynchWithDevice(WCHAR* btNum);
BMS_API DWORD __stdcall BAlertSearchBTDevices(WCHAR* deviceNames,WCHAR* btNum, int* nDeviceNameLength,int& nNumOfDevices);
BMS_API DWORD __stdcall BAlertSearchBTDevicesForCurrentConnection(WCHAR* deviceNames,WCHAR* btNum, int* nDeviceNameLength,int& nNumOfDevices);
BMS_API DWORD __stdcall BAlertSetConfigPath(WCHAR* configPath);
BMS_API DWORD __stdcall BAlertUploadFWToXSeriesHeadset(WCHAR* tchFirmwareName);
BMS_API DWORD __stdcall BAlertUploadFWToOldReciever(WCHAR* tchFirmwareName);
BMS_API DWORD __stdcall BAlertEnumDevicesNoDetails(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p);
BMS_API DWORD __stdcall BAlertEnumDevicesESUDetails(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p);

BMS_API int __stdcall BAlertIsDeviceUSB( BOOL& bUSBDevice );

BMS_API int __stdcall BAlertConfigureFlexAudio(int nAudioOn);

BMS_API DWORD __stdcall BAlertEnumDevicesKeepConnection(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p);

BMS_API int __stdcall BAlertGetFlexDeviceModelForCurrentConnection(int& nDeviceModel, int&, float&);
BMS_API int __stdcall BAlertGetFlexDeviceModelForCurrentConnectionStrip( int& nDeviceModel, int& nDetectedStripCode, float& nPercentageEstimate );
BMS_API int __stdcall BAlertStartAcquisitionForCurrentConnection();
BMS_API int __stdcall BAlertCloseCurrentConnection();
BMS_API DWORD __stdcall BAlertEnumDevicesESUDetailsForCurrentConnection(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p);
BMS_API int __stdcall BAlertFlexMeasureImpedancesForCurrentConnection( int nSensorConfig, int** naImpedances );
BMS_API int __stdcall BAlertGetConnectedDevicePath(WCHAR* chDevicePath);

BMS_API int __stdcall BAlertMakeConnection(WCHAR*);
BMS_API int __stdcall BAlertConnectToPort( WCHAR* chDevicePath );

BMS_API DWORD __stdcall BAlertGetPotentialThirdPartyPorts(int** pPortNumbers,int& nNumOfPorts);

BMS_API int __stdcall BAlertGetBatteryPercOldDevices();

BMS_API void __stdcall BAlertSetNoStripAllowed(BOOL bNoStripAllowed);

BMS_API int __stdcall BAlertSendTPartyTest( int nSerialPort, int nDataLength, unsigned char* pucData );
BMS_API int __stdcall BAlertGetThirdPartyPort( int& nThirdPartyPort );
BMS_API int __stdcall BAlertGetThirdPartyPorts( int& nThirdPartyPorts, int** thirdPartyPorts);

BMS_API BOOL __stdcall BAlertOpenLogFile();
BMS_API void __stdcall BAlertSetDetailedLog(bool);
BMS_API int __stdcall BAlertPingFlexBattery();
BMS_API int __stdcall BAlertPreserveUSC();
BMS_API int __stdcall BAlertGetSTDongleBTInfo(byte& bBTInfo);
BMS_API int __stdcall BAlertGetLastUSCPair(DWORD& dwUSC, DWORD& dwCounter);
BMS_API int __stdcall BAlertSendPCSync( int nComPort );


BMS_API int __stdcall BAlertStartAcquisitionForCurrentConnectionLight();

BMS_API void __stdcall BAlertSetLEHeadset(BOOL bLEDevice);

BMS_API int __stdcall BAlertCheckLEDongleDriver();