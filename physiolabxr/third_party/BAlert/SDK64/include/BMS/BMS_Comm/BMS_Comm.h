
#include "CommLibDefs.h"


//---------------------------------------------------------------------------

/*!
\brief		Enumerates all ports supported by library
\param		pEnumPortsProc Pointer to an application-defined callback function. For more information, see PABMEnumPortsProc.
\param		p Pointer to an application-defined value to be passed to the callback function.
\return		Returns number of supported ports found.
\remark		No remarks.
*/
DWORD BmsCommLibEnumPorts(PABMEnumPortsProc pEnumPortsProc, TABMPortInfoEx* p);

//---------------------------------------------------------------------------

/*!
\brief Connects to a communication port with specified device instance ID
*/
DWORD BmsCommLibConnect(WCHAR* pDeviceInstanceID);
DWORD BmsCommLibConnectToExisting();

/*!
\brief Terminates connection with specified handle
*/
bool  BmsCommLibDisconnect(DWORD dwConnectionHandle);
bool  BmsCommLibClosePort(DWORD dwConnectionHandle);
bool  BmsCommLibReopenPort(DWORD dwConnectionHandle);


//---------------------------------------------------------------------------

/*!
\brief Enumerates all established connections
*/
DWORD BmsCommLibEnumConnections(PABMEnumConnectionsProc pEnumConnectionsProc, void* p);

//---------------------------------------------------------------------------

/*!
\brief Gets information about connection with specified handle
*/
bool  BmsCommLibGetConnectionInfo(DWORD dwConnectionHandle, TABMConnectionInfo* pConnectionInfo);

//---------------------------------------------------------------------------

/*!
\brief Enumerates all ABM devices available on a connection with specified handle
*/
DWORD BmsCommLibEnumDevices(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p, bool, bool);

DWORD BmsCommLibEnumDevicesKeepConnection(DWORD dwConnectionHandle, PABMEnumESUsProc pEnumESUsProc, PABMEnumDAUsProc pEnumDAUsProc, TABMPortInfoEx* p, bool, bool);

//---------------------------------------------------------------------------

/*!
\brief Begins editing settings of external synchronization unit with specified handle
*/
bool  BmsCommLibESUEditSettingsBegin(DWORD dwESUHandle);

/*!
\brief Ends editing settings of external synchronization unit with specified handle
*/
bool  BmsCommLibESUEditSettingsEnd(DWORD dwESUHandle, bool bSaveChanges);

/*!
\brief Sets serial number of external synchronization unit with specified handle
*/
bool  BmsCommLibESUSetSerialNumber(DWORD dwESUHandle, WCHAR SerialNumber[10]);

/*!
\brief Sets Bluetooth port settings of external synchronization unit with specified handle
*/
bool  BmsCommLibESUSetBTPortSettings(DWORD dwESUHandle, TABMESUBTPortSettings* pBTPortSettings, BYTE bPortIndex);

/*!
\brief Sets serial port settings of external synchronization unit with specified handle
*/
bool  BmsCommLibESUSetSERPortSettings(DWORD dwESUHandle, TABMESUSERPortSettings* pSERPortSettings, BYTE bPortIndex);

/*!
\brief Sets parallel port settings of external synchronization unit with specified handle
*/
bool  BmsCommLibESUSetPARPortSettings(DWORD dwESUHandle, TABMESUPARPortSettings* pPARPortSettings);

/*!
\brief Sets debug byte of external synchronization unit with specified handle
*/
bool  BmsCommLibESUSetDebugByte(DWORD dwESUHandle, BYTE bDebugByte);

/*!
\brief Automatically configure external synchronization unit for proper data transfer from data acquisition unit
*/
bool  BmsCommLibESUAutoConfiguration(DWORD dwESUHandle, TABMDAUInfo* pDAUInfo);

//---------------------------------------------------------------------------
bool BmsCommLibPrepareForThirdPartyCheck(DWORD dwESUHandle);

bool BmsCommLibGetThirdPartyCheckStatus(DWORD dwESUHandle);

/*!
\brief Gets information about external synchronization unit on a connection with specified handle
*/
bool  BmsCommLibGetESUInfo(DWORD dwConnectionHandle, bool bReadFromMemory, TABMESUInfo* pESUInfo);

//---------------------------------------------------------------------------

/*!
\brief Begins editing settings of data acquisition unit with specified handle
*/
bool  BmsCommLibDAUEditSettingsBegin(DWORD dwDAUHandle);

/*!
\brief Ends editing settings of data acquisition unit with specified handle
*/
bool  BmsCommLibDAUEditSettingsEnd(DWORD dwDAUHandle, bool bSaveChanges);

/*!
\brief Sets serial number of data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetSerialNumber(DWORD dwDAUHandle, WCHAR SerialNumber[10]);

/*!
\brief Sets number of channels of data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetNumberOfChannels(DWORD dwDAUHandle, BYTE bNumberOfChannels);

/*!
\brief Sets sample size in bytes of data acquisition unit with specified handle.
*/
bool  BmsCommLibDAUSetSampleSizeInBytes(DWORD dwDAUHandle, BYTE bSampleSizeInBytes);

/*!
\brief Sets ECG channel number of data acquisition unit with specified handle.
*/
bool  BmsCommLibDAUSetECGChannel(DWORD dwDAUHandle, BYTE bECGChannel);

// /* !
// \brief DAU configuration function, do not use (internal use only)
// */
// bool  CommLibDAUSetCutOffFrequency(DWORD dwDAUHandle, WORD wCutOffFrequency);
// /* !
// \brief DAU configuration function, do not use (internal use only)
// */
// bool  CommLibDAUSetReconfigurableChannelGains(DWORD dwDAUHandle, TABMReconfigurableChannelGains* pReconfigurableChannelGains);
/*!
\brief Sets tilt transmision of data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetTiltTransmitted(DWORD dwDAUHandle, bool bTiltTransmited);

/*!
\brief Sets BDA number for data acquisition unit with specified handle to connect to
*/
bool  BmsCommLibDAUSetBDA(DWORD dwDAUHandle, WCHAR BDA[13]);

/*!
\brief Sets gain calibration data for data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetGainCalibrationData(DWORD dwDAUHandle, TABMGainCalibrationData* pGainCalibrationData);

/*!
\brief Sets impedance calibration data for data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetImpedanceCalibrationData(DWORD dwDAUHandle, TABMImpedanceCalibrationData* pImpedanceCalibrationData);

/*!
\brief Sets channel pin mapping data for data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetChannelsPinMappingData(DWORD dwDAUHandle, TABMChannelsPinMappingData* pChannelsPinMappingData);

/*!
\brief Sets EEG DC offset for data acquisition unit with specified handle
*/
bool  BmsCommLibDAUSetEEGDCOffsetData(DWORD dwDAUHandle, TABMEEGDCOffsetData* pEEGDCOffsetData);

//---------------------------------------------------------------------------

/*!
\brief Gets information about data acquisition unit with specified handle
*/
bool  BmsCommLibGetDAUInfo(DWORD dwDAUHandle, bool bReadFromMemory, TABMDAUInfo* pDAUInfo);

//---------------------------------------------------------------------------

/*!
\brief Sets working mode of a data acquisition unit with specified handle
*/
bool  BmsCommLibSetWorkingMode(DWORD dwDAUHandle, WORD wNewWorkingMode, WORD* wActualWorkingMode);

/*!
\brief Sets acquisition data block type on a data acquisition unit with specified handle
*/
bool  BmsCommLibSetAcquisitionDataBlockType(DWORD dwDAUHandle, DWORD dwNewDataBlockType, DWORD* dwActualDataBlockType);

/*!
\brief Sets real time clock
*/
bool  BmsCommLibSetRealTimeClock(DWORD dwDAUHandle, BYTE bYear, BYTE bMonth, BYTE bDay, BYTE bHours, BYTE bMinutes, BYTE bSeconds);

//---------------------------------------------------------------------------

/*!
\brief Stops data acquisition on a data acquisition unit with specified handle
*/
bool  BmsCommLibStopAcquisition(DWORD dwDAUHandle);

bool  BmsCommLibPingStartAcquisition(DWORD dwDAUHandle, bool bSaveToSDCard);

/*!
\brief Starts data acquisition on a data acquisition unit with specified handle
*/
bool  BmsCommLibStartAcquisition(DWORD dwDAUHandle, bool bSaveToSDCard);

//---------------------------------------------------------------------------

/*!
\brief Sends bytes to a data acquisition unit with specified handle without waiting for responce
*/
bool  BmsCommLibSendBytesToDAU(DWORD dwDAUHandle,unsigned char* pBytes, int nBytes, int nRetry, int nPause);


//---------------------------------------------------------------------------

/*!
\brief Sets application-defined callback function used for unbuffered output of raw samples from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewRawSample(DWORD dwDAUHandle, PABMOnNewRawSample pOnNewRawSample, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of samples from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewSample(DWORD dwDAUHandle, PABMOnNewSample pOnNewSample, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of accelerometer data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewTilt(DWORD dwDAUHandle, PABMOnNewTilt pOnNewTilt, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of optical channel data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewIRed(DWORD dwDAUHandle, PABMOnNewIRed pOnNewIRed, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of optical channel data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewRed(DWORD dwDAUHandle, PABMOnNewRed pOnNewRed, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of mic data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewMic(DWORD dwDAUHandle, PABMOnNewMicData pOnNewMic, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of mic data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewIRedRaw(DWORD dwDAUHandle, PABMOnNewIRedRawData pOnNewIRedRaw, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of mic data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewTechnicalChannel(DWORD dwDAUHandle, PABMOnNewTechnicalChannel pOnNewTechnicalChannelData, void* p);


/*!
\brief Sets application-defined callback function used for real time notification about new data acquired from slow channels from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewSlowChannels(DWORD dwDAUHandle, PABMOnNewSlowChannels pOnNewSlowChannels, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of timestamps from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewTimestamp(DWORD dwDAUHandle, PABMOnNewTimestamp pOnNewTimestamp, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of third party data from connection where DAU with specified handle is connected
*/
bool  BmsCommLibSetCallbackOnNewThirdPartyData(DWORD dwDAUHandle, PABMOnNewThirdPartyData pOnNewThirdPartyData, void* p);

/*!
\brief Sets application-defined callback function used for unbuffered output of miscellaneous data from data acquisition unit with specified handle
*/
bool  BmsCommLibSetCallbackOnNewProbeData(DWORD dwDAUHandle, PABMOnNewProbeData pOnNewProbeData, void* p);

//---------------------------------------------------------------------------

/*!
\brief Gets number of raw samples pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetRawSamplePending(DWORD dwDAUHandle);

/*!
\brief Gets number of samples pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetSamplePending(DWORD dwDAUHandle);

/*!
\brief Gets number of accelerometer data pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetTiltPending(DWORD dwDAUHandle);

/*!
\brief Gets number of optical channel data pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetIRedPending(DWORD dwDAUHandle);

/*!
\brief Gets number of red pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetRedPending(DWORD dwDAUHandle);

/*!
\brief Gets number of mic pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetMicPending(DWORD dwDAUHandle);

/*!
\brief Gets number of timestamps pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetTimestampPending(DWORD dwDAUHandle);

/*!
\brief Gets number of third party data bytes pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetThirdPartyDataPending(DWORD dwDAUHandle);

/*!
\brief Gets number of misc data pending in buffered output from data acquisition unit with specified handle
*/
DWORD BmsCommLibGetProbeDataPending(DWORD dwDAUHandle);

//---------------------------------------------------------------------------

/*!
\brief Reads samples pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetRawSample(DWORD dwDAUHandle, TABMRawSampleInfo* pRawSampleInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads raw samples pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetSample(DWORD dwDAUHandle, TABMSampleInfo* pSampleInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads accelerometer data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetTilt(DWORD dwDAUHandle, TABMTiltInfo* pTiltInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads optical channel data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetIRed(DWORD dwDAUHandle, TABMIRedInfo* pIRedInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads optical channel data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetRed(DWORD dwDAUHandle, TABMRedInfo* pRedInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads mic data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetMic(DWORD dwDAUHandle, TABMMicInfo* pMicInfo, DWORD dwCount, DWORD* dwActualCount);


/*!
\brief Reads IRedRaw data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetIRedRaw(DWORD dwDAUHandle, TABMIRedRawInfo* pIRedRawInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads Technical Channel data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetTechnicalChannel(DWORD dwDAUHandle, TABMTechnicalChannelInfo* pTechnicalChannelInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads data acquired from slow channels from data acquisition unit with specified handle
*/
bool  BmsCommLibGetSlowChannels(DWORD dwDAUHandle, TABMSlowChannelsInfo* pSlowChannelsInfo);

/*!
\brief Reads timestamps pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetTimestamp(DWORD dwDAUHandle, TABMTimestampInfo* pTimestampInfo, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads third party data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetThirdPartyData(DWORD dwDAUHandle, BYTE* pBuffer, DWORD dwCount, DWORD* dwActualCount);

/*!
\brief Reads probe data pending in buffered output from data acquisition unit with specified handle
*/
bool  BmsCommLibGetProbeData(DWORD dwDAUHandle, TABMProbeDataInfo* pProbeDataInfo, DWORD dwCount, DWORD* dwActualCount);

//---------------------------------------------------------------------------



bool BmsCommLibSetCallbackOnNewTechnicalChannel(DWORD dwDAUHandle, PABMOnNewTechnicalChannel pOnNewTechnicalCh, void* p);

bool BmsCommLibConnectToPort(WCHAR* chDevicepath);


bool BmsCommLibESUQuickQuery(DWORD dwESUHandle, BYTE& bBTInfo, BYTE& bAUXInfo);
bool BmsCommLibPreserveUSC(DWORD dwDAUHandle);