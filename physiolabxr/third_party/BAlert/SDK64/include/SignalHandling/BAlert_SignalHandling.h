//#ifdef BALERTLAB_SIGNALHANDLING_EXPORTS
#define BALERT_SIGNALHANDLING_API __declspec(dllexport)
//#else
//#define BALERTLAB_SIGNALHANDLING_API __declspec(dllimport)
//#endif


#include "TypeDef.h"
#include "ErrorCode.h"
#include "Helper.h"





BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetHeader(WCHAR* pszFileName, _BALERT_LAB_HEADER& header);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_CheckChannelList(WCHAR* pszFileName, WCHAR* chChannelList, int nChannels);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetFileInfo(WCHAR* pszFileName, _BALERT_LAB_FILEINFO& fileInfo);
BALERT_SIGNALHANDLING_API void __stdcall BAlert_Sig_GetLastError(WCHAR** pszErrorMessage);
BALERT_SIGNALHANDLING_API void __stdcall BAlert_Sig_FreeBuffer( void* ptrBuffer);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_RepairFile( WCHAR* pszEdfFileName, WCHAR* pszOutputPath, WCHAR** pszRepairedFileNames );
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetData(WCHAR* pszFileName, WCHAR** chNames, int& nSamples, int& nChannelsAll, 
																		  int& nChannelsInFile, float** fData,
																		  int& nChannelsConstructed, float** fConstructedData, 
																		  int& nTimeChannels, unsigned char** chTimeData);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetEEGFileChannels( WCHAR* pszFileName,WCHAR** pszChannels , int& nChannels, bool bIncludeDeconChannels );
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetConfigurationInfo( WCHAR* pszFileName,WCHAR** chConfigurationName, int& nDeviceTypeCode);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetMissedBlock(WCHAR* pszFileName, int& nSamples, int** nMissedBlock);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetImpMissedBlock(WCHAR* pszFileName, int& nSamples, int** nImpMissedBlock);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetAllMissedBlock(WCHAR* pszFileName, int& nSamples, int** nImpMissedBlock);

BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetChannelsForProcessing(WCHAR* pszFileName,WCHAR** pszChannels , int& nChannels);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetFileChannels( WCHAR* pszFileName, WCHAR** pszChannels , int& nChannels );
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetClassificationChannels( WCHAR* pszFileName, WCHAR** pszChannels , int& nChannels );
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetAvailableChannels( WCHAR* pszFileName, WCHAR** pszChannels , int& nChannels);
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetEyeBlinkChannels( WCHAR* pszFileName, WCHAR** pszChannels, int& nChannels );
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_GetChannelMap( WCHAR* pszFileName, int& nChannelMap );
BALERT_SIGNALHANDLING_API int __stdcall BAlert_Sig_EditFile_CustomOutPath(WCHAR* pszFileName,WCHAR* pszOutFileType, WCHAR* pszOutputPath, WCHAR** pszOutFileName, 
																  int nStartEp, int nStopEp, int nStartSample, int nStopSample, WCHAR* chChannels, int& nChannels);