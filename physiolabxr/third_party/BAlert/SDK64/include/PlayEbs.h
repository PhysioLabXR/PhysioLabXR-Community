
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the PLAYEBS_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// PLAYEBS_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef PLAYEBS_EXPORTS
#define PLAYEBS_API __declspec(dllexport)
#else
#define PLAYEBS_API __declspec(dllimport)
#endif

#include "..\include\TypeDef.h"



//PLAYEBS_API BOOL Open( TCHAR* sFullPath, TCHAR* definitionfile );

PLAYEBS_API BOOL OpenCustom( TCHAR* sFullPath, _SESSION_INFO* pSessionInfo, int nDeviceType, BOOL bHRVAnalysis );

PLAYEBS_API _SESSION_INFO_PE*  __stdcall  GetSessionInfo();

PLAYEBS_API BOOL PlayFile(double nPauseInMiliseconds = 78.125 );

PLAYEBS_API void ClosePlay();



//get functions
PLAYEBS_API _BRAIN_STATE*  __stdcall  GetBrainStatePE(int& );

PLAYEBS_API float*  __stdcall  GetRawDataPE(int&);

PLAYEBS_API float*  __stdcall  GetRawRawDataPE(int&);

PLAYEBS_API float*  __stdcall  GetFilteredDataPE(int&);

PLAYEBS_API float*  __stdcall  GetDeconDataPE(int& );

PLAYEBS_API float*  __stdcall  GetQualityChannelDataPE(int& );

PLAYEBS_API float*  __stdcall  GetPSDDataPE(int& );

PLAYEBS_API float*  __stdcall  GetPSDDatarawPE(int& );

PLAYEBS_API float*  __stdcall  GetEKGDataPE(int& );



PLAYEBS_API float*  __stdcall  GetMovementDataPE(int& );

PLAYEBS_API float*  __stdcall  GetRawTiltDataPE(int& );

PLAYEBS_API float*  __stdcall  GetAnglesDataPE(int& );

PLAYEBS_API int  __stdcall  InitZScoreDataPE( TCHAR* pchZScoreSourceList );

PLAYEBS_API float*  __stdcall  GetZScoreDataPE(int&, int& );







PLAYEBS_API int   __stdcall  SetArtifactsCallbackFuncsPE(void (__stdcall *pFuncEB)(int epstart, int offstart,  float shour, float sminute, float ssecond, float smili , int epend, int offend,  float ehour, float eminute, float esecond, float emili), 
													  void (__stdcall *pFuncExc)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
													  void (__stdcall *pFuncSat)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
													  void (__stdcall *pFuncSpk)(int indexch, int epstart,  int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili),
													  void (__stdcall *pFuncEMG)(int indexch,int epstart, int offstart,  float shour, float sminute, float ssecond, float smili, int epend, int offend, float ehour, float eminute, float esecond, float emili));

PLAYEBS_API float*  __stdcall  GetCurrentQualityDataPE(int& );

PLAYEBS_API float*  __stdcall  GetClassQualityDataPE(int& );

PLAYEBS_API int __stdcall GetPEChannelsInfo( _EEGCHANNELS_INFO & stEEGChannelsInfo);
PLAYEBS_API int __stdcall GetPEDeviceInfo(char* sFullPath,int& nDeviceType);
PLAYEBS_API int __stdcall GetPEPacketChannelNmbInfo(int& nRawPacketChannelsNmb, int& nDeconPacketChannelsNmb,  int& nPSDPacketChannelsNmb, int& nRawPSDPacketChannelNmb, int& nQualityPacketChannelNmb);
PLAYEBS_API int __stdcall GetPEBandsDescription(char* pszPSDBands, int& count);


PLAYEBS_API float*  __stdcall  GetPSDBandwidthRawDataPE(int& nCountPackages, int& nPackageSize);
PLAYEBS_API float*  __stdcall  GetPSDBandwidthClassDataPE(int& nCountPackages, int& nPackageSize);
PLAYEBS_API float*  __stdcall  GetBandOverallPSDRawDataPE(int& nCountPackages, int& nPackageSize);
PLAYEBS_API float*  __stdcall  GetBandOverallPSDClassDataPE(int& nCountPackages, int& nPackageSize);

