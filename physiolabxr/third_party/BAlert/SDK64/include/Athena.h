// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the FFFFFF_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// FFFFFF_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef ATHENA_EXPORTS
#define ATHENA_API __declspec(dllexport)
#else
#define ATHENA_API __declspec(dllimport)
#endif

#include "TypeDef.h"

ATHENA_API int  __stdcall  StartAcquisition();
ATHENA_API int  __stdcall  CloseCurrentConnection();

 
ATHENA_API int  __stdcall  StopAcquisition();
 
ATHENA_API int  __stdcall  InitSessionForCurrentConnection(int DeviceType,int SessionType,int nSelectedDeviceHandle, BOOL bPlayEBS );
ATHENA_API int  __stdcall  InitSessionForCurrentConnectionWithHRV(int DeviceType, int SessionType, int nSelectedDeviceHandle, BOOL bPlayEBS, BOOL bHRVAnalysisInSession);


ATHENA_API _BRAIN_STATE*  __stdcall  GetBrainState(int& ); 

ATHENA_API float*  __stdcall  GetRawData(int&);

ATHENA_API float*  __stdcall  GetDeconData(int& );

ATHENA_API float*  __stdcall  GetQualityChannelData(int& );

ATHENA_API float*  __stdcall  GetPSDData(int& );





ATHENA_API float*  __stdcall  GetMovementData(int& );

ATHENA_API float*  __stdcall  GetPSDDataraw(int& );

ATHENA_API float*  __stdcall  GetBandOverallPSDData(int& nCountPackages, int& nPackageSize);

ATHENA_API float*  __stdcall  GetBandOverallPSDRawData(int& nCountPackages, int& nPackageSize);

ATHENA_API float*  __stdcall  GetPSDBandwidthData(int& nCountPackages, int& nPackageSize);

ATHENA_API float*  __stdcall  GetPSDBandwidthRawData(int& nCountPackages, int& nPackageSize);

ATHENA_API int __stdcall  GetPSDCountBands(int& nCountBandwidths, int& nCountBandwidthsOverall );

ATHENA_API int  __stdcall  InitZScoreData( wchar_t* sZScoreSourceList );

ATHENA_API int  __stdcall  ResetZScoreData( wchar_t* sNewZScoreSourceList );

ATHENA_API int  __stdcall  ReleaseZScoreData();

ATHENA_API float*  __stdcall  GetZScoreData(int& nCountPackages, int& nPackageSize);

ATHENA_API _ESU_PORT_INFO* __stdcall  GetESUPortInfo();

ATHENA_API int  __stdcall SaveESUPortInfo(_ESU_PORT_INFO* pESUPortInfo);

ATHENA_API int   __stdcall  TechnicalMonitoring(void (__stdcall *pFunc)(CHANNEL_INFO*, int&), int nSeconds, _DEVICE_INFO* pDeviceInfo = NULL);





ATHENA_API unsigned char*  __stdcall  GetThirdPartyData(int& );

ATHENA_API unsigned char*  __stdcall  GetTimeStampsStreamData(int nType);

ATHENA_API int   __stdcall  SetArtifactsCallbackFuncs(void (__stdcall *pFuncEB)(int epstart, int offstart,  float shour, float sminute, float ssecond, float smili , int epend, int offend,  float ehour, float eminute, float esecond, float emili), 
															  void (__stdcall *pFuncExc)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
															  void (__stdcall *pFuncSat)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
															  void (__stdcall *pFuncSpk)(int indexch, int epstart,  int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
															  void (__stdcall *pFuncEMG)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili));

ATHENA_API int __stdcall RegisterCallbackOnError(void (__stdcall *pFunc)(int));

ATHENA_API int __stdcall RegisterCallbackOnStatusInfo(void (__stdcall *pFunc)(_STATUS_INFO*));

ATHENA_API int __stdcall RegisterCallbackOnThirdParty(void (__stdcall *pFunc)(const unsigned char*,int));

ATHENA_API int __stdcall RegisterCallbackDataArrived(void (__stdcall *pFunc)(int));

ATHENA_API int __stdcall RegisterCallbackMissedBlocks(void (__stdcall *pFunc)(int, int));

ATHENA_API int __stdcall RegisterCallbackDeviceDetectionInfo(void (__stdcall *pFunc)(wchar_t*, int));

ATHENA_API int __stdcall StopTechnicalMonitoring();

ATHENA_API float*  __stdcall  GetEKGData(int& );







ATHENA_API int  __stdcall  SetMarker(int epoch, int offset, int val);

ATHENA_API int  __stdcall  SetMarkerWithComment(int epoch, int offset, int val, wchar_t* comment);

ATHENA_API int  __stdcall  GetFirmwareVer(_BALERT_FIRMWARE_VERSION &stFWVer);

//play ebs mode functions

ATHENA_API BOOL __stdcall PlayInEbsModeDataAll(float* pEegData, int nSamples, WORD* pOpticalData, int nOpticalSamples, int* nTiltData, int nTiltSamples, 
											    WORD* pOpticalRawData, int nOpticalRawSamples );


ATHENA_API _EXTERN_CALC_TABLE_NAME*   __stdcall  GetMapCalcChannels( int& nSize );

ATHENA_API void    __stdcall  StopPlayEbsData();
 

ATHENA_API float* __stdcall  GetQualityChannelData(int& nCount);



ATHENA_API int __stdcall GetCurrentSDKMode();
ATHENA_API int __stdcall GetMissedBlocks();

ATHENA_API int  __stdcall  InitiatePlaybackSession( _SESSION_INFO* pSessionInfo, int nDeviceType, _DEVICE_INFO* pDeviceInfo, int nHardwareDeviceType, BOOL bTilt, BOOL bHRVAnalysis);

ATHENA_API float*  __stdcall  GetRawRawData(int&);

ATHENA_API float*  __stdcall  GetFilteredData(int&);

ATHENA_API float* __stdcall  GetTiltRawData(int& nCount);

ATHENA_API float* __stdcall  GetTiltAnglesData(int& nCount);

ATHENA_API int __stdcall SetDefinitionFile(wchar_t* pDefinitionFile);

ATHENA_API int __stdcall SetDestinationFile(wchar_t* pDestinationFile);





ATHENA_API int __stdcall GetBatteryLevel(float &fBatteryLevel, int &nBatteryPercentage);

ATHENA_API int __stdcall GetPacketChannelNmbInfo(int& nRawPacketChannelsNmb, int& nDeconPacketChannelsNmb,  int& nPSDPacketChannelsNmb, int& nRawPSDPacketChannelNmb, int& nQualityPacketChannelNmb);

ATHENA_API int __stdcall GetChannelMapInfo( _CHANNELMAP_INFO & stChannelMapInfo);

ATHENA_API int __stdcall GetEEGChannelsInfo( _EEGCHANNELS_INFO & stEEGChannelsInfo);

ATHENA_API int __stdcall GetTimestampType();

ATHENA_API int __stdcall SetConfigPath( wchar_t* pConfigPath );

ATHENA_API int __stdcall GetThirdPartyTimestamp(int &label, int &millisESU, _SYSTEMTIME &st);

ATHENA_API float* __stdcall  GetCurrentQualityData(int& nCount);

ATHENA_API float* __stdcall  GetClassQualityData(int& nCount);

ATHENA_API int  __stdcall GetBandsDescription(wchar_t** pszBands, int& count);

ATHENA_API int  __stdcall  InitiatePESession(int nDeviceType, int sessionType, wchar_t* chDefFile, BOOL bTilt);

ATHENA_API void __stdcall FreeBuffer( void* ptrBuffer);

ATHENA_API int __stdcall  MeasureImpedances(void(__stdcall* pFunc)(ELECTRODE* pEl, int& nCount));

ATHENA_API int  __stdcall GetOverallBandsDescription(wchar_t** pszBands, int& count);

ATHENA_API void  __stdcall InitOCEANAcquisition(BOOL bStartOCEAN);

ATHENA_API int  __stdcall GetEdfId( char** pszEdfId );

ATHENA_API void  __stdcall SetSwPackageVersion( wchar_t* pszVersion );

ATHENA_API int  __stdcall GetDestinationFile( wchar_t** pszDestinationFile );

ATHENA_API int __stdcall  InitSessionStream(int nDeviceType,int nSessionType,int nSelectedDeviceHandle, BOOL bPlayEBS);

ATHENA_API int __stdcall  StartAcquisitionStream( char* pchIPAddress );

ATHENA_API int __stdcall  StopAcquisitionStream();

ATHENA_API int   __stdcall  SetMBCallbackFuncs(void (__stdcall *pFuncMB)(int epstart, int offstart,  float shour, float sminute, float ssecond, float smili , int epend, int offend,  float ehour, float eminute, float esecond, float emili));



ATHENA_API int   __stdcall  SetPeriodicImpedanceMonitor(int nPeriodeMinutes);

ATHENA_API int   __stdcall  SetPeriodicImpMBCallbackFuncs(void (__stdcall *pFuncMB)(int epstart, int offstart,  float shour, float sminute, float ssecond, float smili , int epend, int offend,  float ehour, float eminute, float esecond, float emili));

ATHENA_API BOOL   __stdcall  CheckImpedancesPerRequest();

ATHENA_API void __stdcall SetImpPerRequestCallbackFuncs(void (__stdcall *pFunc)(int bImpedanceStatus));

ATHENA_API void  __stdcall GetEDFDestinationFile( wchar_t** pszDestinationFile );



ATHENA_API _DEVICE_INFO_DETAILS* __stdcall GetDeviceInfoDetails( int& nErrorCode, int& nStripCode );

ATHENA_API int  __stdcall  CheckForLongMB(BOOL& bLongMBHappened);
ATHENA_API void __stdcall FreeNonArrayBuffer( void* ptrBuffer);

ATHENA_API _DEVICE_INFO* __stdcall  GetDeviceInfo(int nRetry /*default 20*/, int& nStripCode, int& nErrorCode);
ATHENA_API BOOL  __stdcall  IsConnectionOpened();
ATHENA_API int  __stdcall GetDetectedDevicePath( wchar_t** pszDevicePath );



ATHENA_API _ESU_PORT_INFO* __stdcall  GetESUPortInfoForCurrentConnection();

ATHENA_API void  __stdcall SetNoStripAllowed(BOOL bNoStripAllowed);
ATHENA_API void  __stdcall SetClassicFlexOldFwAllowed(BOOL bOldClassicFwAllowed);

ATHENA_API BOOL __stdcall  GetESUSerialNumber(wchar_t** esuSerialNum);
ATHENA_API BOOL __stdcall  GetDeviceSerialNumber(wchar_t** deviceSerialNum);

ATHENA_API int  __stdcall GetMachineGuidBegin(wchar_t** ptchMachineGuidBegin);
ATHENA_API int __stdcall PingFlexBattery();
ATHENA_API int __stdcall GetReceiverInfo();
ATHENA_API int  __stdcall StartLSLReceiver(BOOL bStartOCEANReceiver, BOOL bStartReceiver);
ATHENA_API int __stdcall  InvokeResynchingProcedure();
ATHENA_API void __stdcall SetSignalCheckDestinationFile( bool bSet );

ATHENA_API void __stdcall SetEEGLSLSettings(_LSLSTREAMING_INFO* info);
ATHENA_API void  __stdcall EnableHRVAnalysis(BOOL bEnableHRV);

ATHENA_API void  __stdcall SetTeaming(wchar_t* teamingID);
