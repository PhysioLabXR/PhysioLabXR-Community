
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the ABMTNGRMDLL_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// ABMTNGRMDLL_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.
#ifdef ABMTNGRMDLL_EXPORTS
#define ABMTNGRMDLL_API __declspec(dllexport)
#else
#define ABMTNGRMDLL_API __declspec(dllimport)
#endif

#include "..\TypeDefDataTypes.h"

//Error codes
#define     ABM_DS_NOT_USE_METHOD					 0
#define     ABM_DS_SUCCESS							 1
#define     ABM_DS_ERROR_INIT_PARAMETER				-1
#define     ABM_DS_ERROR_INIT_WINSOCK2				-2
#define     ABM_DS_ERROR_CREATE_SOCKET				-3
#define     ABM_DS_ERROR_GET_HOST					-4
#define     ABM_DS_ERROR_CONNECT					-5
#define     ABM_DS_ERROR_NON_BLOCK_MODE				-6
#define		ABM_DS_ERROR_BAD_HANDLE					-7
#define     ABM_DS_ERROR_TO_MANY_CONNECTIONS		-8
#define     ABM_DS_ERROR_WRONG_INPUT_PARAMETER		-9
#define     ABM_DS_ERROR_NONEXISTANT_CONNECTION		-10
#define     ABM_DS_ERROR_CHANNELMAP_NOT_SET			-11
#define     ABM_DS_ERROR_INITIALIZATION_FAILED		-12
#define     ABM_DS_ERROR_UKNOWN_PROPERTY			-13

#define     ABM_DS_CLIENT_TCP_VIEW						1
#define     ABM_DS_CLIENT_UDP							0
#define     ABM_DS_CLIENT_TCP_CONTROL					2

#define     ABM_DS_PROPERTY_BEACON_INTERVAL			250


// connection control
ABMTNGRMDLL_API long  __stdcall OpenConnection(char* sInput, int &handle,int bTCP = ABM_DS_CLIENT_TCP_VIEW);
ABMTNGRMDLL_API long  __stdcall CloseConnection(int handle);

ABMTNGRMDLL_API int _stdcall  GetIPInfo(int &IP0, int &IP1, int &IP2, int &IP3, int &nPort, int &nLocalPort, int handle);
ABMTNGRMDLL_API int __stdcall GetNumberOfConnections();

ABMTNGRMDLL_API _ABM_DATA_DEVICE_INFO*  __stdcall  AgetDeviceInfo(int handle);
ABMTNGRMDLL_API int   __stdcall     AgetProtocolType(int handle);
ABMTNGRMDLL_API char*   __stdcall   AgetChannelMapInfo(int &nLength, int handle);

ABMTNGRMDLL_API float*  __stdcall   AgetNotification(int& nCount, int handle );
ABMTNGRMDLL_API int		__stdcall   ASendNotification( float* fNotification, int handle );
ABMTNGRMDLL_API int		__stdcall   AGetBattery(int& nEpoch, int handle);
ABMTNGRMDLL_API int		__stdcall   AGetImpedances(int& nEpoch, int& nImpOnLineStatus, int nImpOnLineValues[24], int handle);
ABMTNGRMDLL_API int		__stdcall   AgetLastError(int& nEpoch, int handle);
ABMTNGRMDLL_API int		__stdcall   AgetMissedBlocks(int& nEpoch, int handle);
ABMTNGRMDLL_API int		__stdcall   ASendCommand(int nCommandType, int nCommandSubType, int nCommandSequence, 
												 int param0, int param1, int nHandle);
ABMTNGRMDLL_API float*  __stdcall   AgetResponse(int& nCount, int handle);


// get data - fixed size
ABMTNGRMDLL_API float*	__stdcall	AgetCWPC(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetEKG(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetMovement(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall   AgetPulseRate(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall   AgetHaptic(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetACC(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetAngles(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall   AgetRawOptical(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetQuality(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetZScore(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetBandOverallPSDData(int& nCount, int handle);  
ABMTNGRMDLL_API float*  __stdcall	AgetBandOverallPSDRawData(int& nCount, int handle);  
ABMTNGRMDLL_API float*  __stdcall	AgetPSDBandwidthData(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetPSDBandwidthRawData(int& nCount, int handle);

// get data - variable size
ABMTNGRMDLL_API float*  __stdcall	AgetDecon(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetRaw(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetPSD(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetPSDraw(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetQualityChannel(int& nCount, int handle);


ABMTNGRMDLL_API long  __stdcall  GetTimeStamp(float& hour, float& minute, float& seconds, float& miliseconds, int handle);
ABMTNGRMDLL_API long  __stdcall  GetEbsTimeStamp(int& nEpoch, int& nOffset, int handle);
ABMTNGRMDLL_API char*  __stdcall AgetThirdParty( int& nCount, int handle );

ABMTNGRMDLL_API int  __stdcall ASetPropertyValue(int nPropertyID, int propertyValue, int handle);

// data calbacks
ABMTNGRMDLL_API int   __stdcall		ASetArtifactsCallbackFuncs(void (__stdcall *pFuncEB)(int epstart, int offstart,  float shour, float sminute, float ssecond, float smili , int epend, int offend,  float ehour, float eminute, float esecond, float emili), 
															   void (__stdcall *pFuncExc)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
															   void (__stdcall *pFuncSat)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
															   void (__stdcall *pFuncSpk)(int indexch, int epstart,  int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
															  void (__stdcall *pFuncEMG)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili), int handle);
ABMTNGRMDLL_API int    __stdcall	ASetCheckImpedances(void (__stdcall *pFunc)(_ABM_DATA_ELECTRODE*, int&), int handle);

ABMTNGRMDLL_API float*  __stdcall	AgetCurrentQuality(int& nCount, int handle);
ABMTNGRMDLL_API float*  __stdcall	AgetClassQuality(int& nCount, int handle);