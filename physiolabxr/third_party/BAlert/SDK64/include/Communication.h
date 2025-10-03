#ifndef _COMMUNICATION_LIB_H_
#define _COMMUNICATION_LIB_H_

#pragma once
#include "..\include\TCP\ThirdPartyCommunication.h"

static HINSTANCE hComLib;

static int(__stdcall* STARTSERVER)(_SERVER_INFO);
#define StartServer(server) ((*STARTSERVER)(server))

static int (__stdcall *STOPSERVER)(void);
#define StopServer() ((*STOPSERVER)())

static int (__stdcall *SENDDATAGRAM)(char*, int, char*);
#define SendDatagram(pData, nLength, messageCode) ((*SENDDATAGRAM)(pData, nLength, messageCode))

static int (__stdcall *SENDRAWDATA)(float*, int);
#define SendRawData(pData, nLength) ((*SENDRAWDATA)(pData, nLength))

static int (__stdcall *SENDDECONDATA)(float*, int);
#define SendDeconData(pData, nLength) ((*SENDDECONDATA)(pData, nLength))

static int (__stdcall *SENDQUALITY)(float*, int);
#define SendQuality(pData, nLength) ((*SENDQUALITY)(pData, nLength))

static int (__stdcall *SENDCURRENTQUALITY)(float*, int);
#define SendCurrentQuality(pData, nLength) ((*SENDCURRENTQUALITY)(pData, nLength))

static int (__stdcall *SENDCLASSQUALITY)(float*, int);
#define SendClassQuality(pData, nLength) ((*SENDCLASSQUALITY)(pData, nLength))

static int (__stdcall *SENDQUALITYCHANNEL)(float*, int);
#define SendQualityChannel(pData, nLength) ((*SENDQUALITYCHANNEL)(pData, nLength))

static int (__stdcall *SENDPSDDATA)(float*, int);
#define SendPSDData(pData, nLength) ((*SENDPSDDATA)(pData, nLength))

static int (__stdcall *SENDPSDRAWDATA)(float*, int);
#define SendPSDRawData(pData, nLength) ((*SENDPSDRAWDATA)(pData, nLength))

static int (__stdcall *SENDEBDATA)(char*, int);
#define SendEBData(pData, nLength) ((*SENDEBDATA)(pData, nLength))

static int (__stdcall *SENDEXCDATA)(char*, int);
#define SendEXCData(pData, nLength) ((*SENDEXCDATA)(pData, nLength))

static int (__stdcall *SENDEMGDATA)(char*, int);
#define SendEMGData(pData, nLength) ((*SENDEMGDATA)(pData, nLength))

static int (__stdcall *SENDSATDATA)(char*, int);
#define SendSATData(pData, nLength) ((*SENDSATDATA)(pData, nLength))

static int (__stdcall *SENDSPKDATA)(char*, int);
#define SendSPKData(pData, nLength) ((*SENDSPKDATA)(pData, nLength))

static int (__stdcall *SENDEKGDATA)(float*, int);
#define SendEKGData(pData, nLength) ((*SENDEKGDATA)(pData, nLength))

static int (__stdcall *SENDBRAINSTATE)(float*, int);
#define SendBrainState(pData, nLength) ((*SENDBRAINSTATE)(pData, nLength))

//static int (__stdcall *SENDBATTERYPERCENTAGE)(int, int);
//#define SendBatteryPercentage(nBatteryPercentage, nEpoch) ((*SENDBATTERYPERCENTAGE)(nBatteryPercentage, nEpoch))

static int (__stdcall *SENDBATTERYANDIMPENDACES)(int, int, int, int[] );
#define SendBatteryAndImpedances(nBatteryPercentage, nEpoch, nImpOnlineStatus, nImpOnlineValues) ((*SENDBATTERYANDIMPENDACES)(nBatteryPercentage, nEpoch, nImpOnlineStatus, nImpOnlineValues))

static int (__stdcall *SENDMISSEDBLOCKS)(int, int);
#define SendMissedBlocks(nMissebBlocks, nEpoch) ((*SENDMISSEDBLOCKS)(nMissebBlocks, nEpoch))



static int (__stdcall *STOPSERVERTCP)(void);
#define StopServerTCP() ((*STOPSERVERTCP)())

static int (__stdcall *STOPSERVERUDP)(void);
#define StopServerUDP() ((*STOPSERVERUDP)())

static int (__stdcall *GETSTATUSSERVERTCP)(unsigned int&);
#define GetStatusServerTCP(clients) ((*GETSTATUSSERVERTCP)(clients))

static int (__stdcall *GETSTATUSSERVERUDP)(unsigned int&);
#define GetStatusServerUDP(clients) ((*GETSTATUSSERVERUDP)(clients))

static int (__stdcall *SENDUDPDATASTREAMINGTS)(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
#define SendUDPDatastreamingTS(epoch, offset, hour, minute, second, millisecond) ((*SENDUDPDATASTREAMINGTS)(epoch, offset, hour, minute, second, millisecond))

static int (__stdcall *SENDDATAGRAMTCP)(char*, int, char*);
#define SendDatagramTCP(pData, nLength, messageCode) ((*SENDDATAGRAMTCP)(pData, nLength, messageCode))


static int (__stdcall *SETEEGCHANNELMAPINFO)(char*,int);
#define SetEEGChannelMapInfo(chMapInfoBytes,nLength) ((*SETEEGCHANNELMAPINFO)(chMapInfoBytes,nLength))


static int (__stdcall *SENDTCPTIMESTAMP)(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
#define SendTCPTimeStamp(epoch, offset, hour, minute, second, millisecond) ((*SENDTCPTIMESTAMP)(epoch, offset, hour, minute, second, millisecond))

static float* (__stdcall *GETNOTIFICATIONS)(int&);
#define GetNotifications(nCount) ((*GETNOTIFICATIONS)(nCount))

static int (__stdcall *SENDNOTIFICATION)(float*, int);
#define SendNotification(pData, nLength) ((*SENDNOTIFICATION)(pData, nLength))

static int (__stdcall *SENDMOVEMENT)(float*, int);
#define SendMovement(pData, nLength) ((*SENDMOVEMENT)(pData, nLength))

static int (__stdcall *SENDRAWTILTS)(float*, int);
#define SendRawTilts(pData, nLength) ((*SENDRAWTILTS)(pData, nLength))

static int (__stdcall *SENDANGLES)(float*, int);
#define SendAngles(pData, nLength) ((*SENDANGLES)(pData, nLength))

static int (__stdcall *SENDZSCORE)(float*, int);
#define SendZScore(pData, nLength) ((*SENDZSCORE)(pData, nLength))

static int (__stdcall *SENDBANDOVERALLPSDDATA)(float*, int);
#define SendBandOverallPSDData(pData, nLength) ((*SENDBANDOVERALLPSDDATA)(pData, nLength))

static int (__stdcall *SENDBANDOVERALLPSDRAWDATA)(float*, int);
#define SendBandOverallPSDRawData(pData, nLength) ((*SENDBANDOVERALLPSDRAWDATA)(pData, nLength))

static int (__stdcall *SENDPSDBANDWIDTHDATA)(float*, int);
#define SendPSDBandwidthData(pData, nLength) ((*SENDPSDBANDWIDTHDATA)(pData, nLength))

static int (__stdcall *SENDPSDBANDWIDTHRAWDATA)(float*, int);
#define SendPSDBandwidthRawData(pData, nLength) ((*SENDPSDBANDWIDTHRAWDATA)(pData, nLength))

static int (__stdcall *SENDPULSERATEDATA)(float*, int);
#define SendPulseRateData(pData, nLength) ((*SENDPULSERATEDATA)(pData, nLength))

//static int (__stdcall *GETDEBUGINFOTCP)(_TCP_DEBUG_INFO* pTCPDebugInfo);
//#define GetDebugInfoTCP(pTcpDebugInfo) ((*GETDEBUGINFOTCP)(pTCPDebugInfo))

static int (__stdcall *SENDHAPTICDATA)(float*, int);
#define SendHapticData(pData, nLength) ((*SENDHAPTICDATA)(pData, nLength))

static int (__stdcall *SENDRAWOPTICALDATA)(float*, int);
#define SendRawOpticalData(pData, nLength) ((*SENDRAWOPTICALDATA)(pData, nLength))

static int (__stdcall *SENDTHIRDPARTYDATA)(char*, int); 
#define SendThirdParty(pData, nLength) ((*SENDTHIRDPARTYDATA)(pData, nLength))

static int (__stdcall *SENDIMPEDANCE)(char*, int); 
#define SendImpedance(pData, nLength) ((*SENDIMPEDANCE)(pData, nLength))


///////////////////////////////////////////////////////////////////////////////
// DLL linkage functions (temporary)
static HINSTANCE OpenComLib(void)
{

#ifdef _WIN64
	//hComLib = LoadLibrary(_T("c:\\ABM\\B-Alert\\SDK64\\bin\\ABM_ThirdPartyCommunication64.dll"));
	hComLib = LoadLibrary(_T("ABM_ThirdPartyCommunication64.dll"));
#else
	hComLib = LoadLibrary(_T("c:\\ABM\\B-Alert\\SDK\\bin\\ABM_ThirdPartyCommunication.dll"));
#endif

	if (hComLib   != NULL)
	{
		STARTSERVER = (int(__stdcall *)(_SERVER_INFO))GetProcAddress(hComLib, "StartServer");
		STOPSERVER = (int(__stdcall *)(void))GetProcAddress(hComLib, "StopServer");
		SENDDATAGRAM = (int(__stdcall *)(char*, int, char*))GetProcAddress(hComLib, "SendDatagram");
		SENDRAWDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendRawData");
		SENDDECONDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendDeconData");
		SENDQUALITY = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendQuality");
		SENDCURRENTQUALITY = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendCurrentQuality");
		SENDCLASSQUALITY = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendClassQuality");

		SENDQUALITYCHANNEL = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendQualityChannel");
		SENDPSDDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendPSDData");
		SENDPSDRAWDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendPSDRawData");
		SENDEBDATA = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendEBData");
		SENDEXCDATA = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendEXCData");
		SENDEMGDATA = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendEMGData");
		SENDSATDATA = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendSATData");
		SENDSPKDATA = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendSPKData");
		SENDEKGDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendEKGData");
		SENDBRAINSTATE = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendBrainState");
		SENDTHIRDPARTYDATA = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendThirdPartyData");
		STOPSERVERTCP = (int(__stdcall *)(void))GetProcAddress(hComLib, "StopServerTCP");
		STOPSERVERUDP = (int(__stdcall *)(void))GetProcAddress(hComLib, "StopServerUDP");
		GETSTATUSSERVERTCP = (int(__stdcall *)(unsigned int&))GetProcAddress(hComLib, "GetStatusServerTCP");
		GETSTATUSSERVERUDP = (int(__stdcall *)(unsigned int&))GetProcAddress(hComLib, "GetStatusServerUDP");
		SENDUDPDATASTREAMINGTS = (int(__stdcall *)(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int))GetProcAddress(hComLib, "SendUDPDatastreamingTS");
		SENDDATAGRAMTCP = (int(__stdcall *)(char*, int, char*))GetProcAddress(hComLib, "SendDatagramTCP");
		SENDTCPTIMESTAMP = (int(__stdcall *)(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int))GetProcAddress(hComLib, "SendTCPTimeStamp");
		SENDMOVEMENT = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendMovement");
		SENDRAWTILTS = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendAccelerometer");
		SENDANGLES = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendAngles");		
		SENDZSCORE = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendZScore");
		SENDBANDOVERALLPSDDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendBandOverallPSDData");
		SENDBANDOVERALLPSDRAWDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendBandOverallPSDRawData");
		SENDPSDBANDWIDTHDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendPSDBandwidthData");
		SENDPSDBANDWIDTHRAWDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendPSDBandwidthRawData");	
		SENDPULSERATEDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendPulseRateData");
		SENDHAPTICDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendHapticData");
		SENDRAWOPTICALDATA = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendRawOpticalData");
		//GETDEBUGINFOTCP=  (int (__stdcall*)(_TCP_DEBUG_INFO*)) GetProcAddress(hComLib, "GetDebugInfoTCP");
		//SENDBATTERYPERCENTAGE =  (int (__stdcall*)(int, int)) GetProcAddress(hComLib, "SendBatteryPercentage");
		SENDBATTERYANDIMPENDACES =  (int (__stdcall*)(int, int, int, int[])) GetProcAddress(hComLib, "SendBatteryAndImpedances");
		SENDMISSEDBLOCKS =   (int (__stdcall*)(int, int)) GetProcAddress(hComLib, "SendMissedBlocks");
		SETEEGCHANNELMAPINFO = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SetEEGChannelMapInfo");
		SENDNOTIFICATION = (int(__stdcall *)(float*, int))GetProcAddress(hComLib, "SendNotification");
		GETNOTIFICATIONS = (float*(__stdcall *)(int&))GetProcAddress(hComLib, "GetNotifications");
		SENDIMPEDANCE = (int(__stdcall *)(char*, int))GetProcAddress(hComLib, "SendImpedance");

		 
	}

	if (!hComLib)
	{
		DWORD dwError = GetLastError();
		LPVOID lpMsgBuf;
		FormatMessage( 
			FORMAT_MESSAGE_ALLOCATE_BUFFER | 
			FORMAT_MESSAGE_FROM_SYSTEM | 
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			dwError,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
			(LPTSTR) &lpMsgBuf,
			0,
			NULL 
			);
		// Display the string.
		::MessageBox( NULL, (LPCTSTR)lpMsgBuf, _T("Error"), MB_OK | MB_ICONINFORMATION );
		// Free the buffer.
		LocalFree( lpMsgBuf );
	}

	return hComLib;
}

static BOOL CloseComLib(void)
{
	BOOL bRet = FreeLibrary(hComLib);
	hComLib = NULL;

	return bRet;
}

#endif

