#ifndef __THIRDPARTYCOMMUNICATION__
#define __THIRDPARTYCOMMUNICATION__

// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the THIRDPARTYCOMMUNICATION_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// THIRDPARTYCOMMUNICATION_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef THIRDPARTYCOMMUNICATION_EXPORTS       
#define THIRDPARTYCOMMUNICATION_API __declspec(dllexport)
#else
#define THIRDPARTYCOMMUNICATION_API __declspec(dllimport)
#endif


// Constants
#define		ABM_3RD_MAX_NUM_READERS 1000
#define		ABM_3RD_SESSION_ID_LENGTH	9
// IP protocol
#define ABM_3RD_PROTOCOL_TRANSPORT_CONTROL  1
#define ABM_3RD_PROTOCOL_USER_DATAGRAM      2
// protocol on top of streaming protocol (PROTOCOL_TRANSPORT_CONTROL  1)
#define ABM_3RD_DATASTREAMING_PROTOCOL  1
#define ABM_3RD_MINIMAL_PROTOCOL        2



// ABM 3RD Structs
typedef struct TD_SERVER_INFO{
	unsigned int  	port; //server port to listen to
	unsigned char  	chProtocolTCPIP; //wheather to use TCP or UDP on given port
	unsigned char     chDatagramProtocol; // Protocol above TCP/UDP (i.e. DATASTREAMING_PROTOCOL)
	// 1 - no info 	// 10 - raw, 11 - raw+decon, 12 - raw+decon+class, 13 - raw+decon+class+workload

	 int nDatastreaming_RawChannels;
	 int nDatastreaming_EKGindex;
	 int nPSDBandsCount;
	 int nPSDBandsOverallCount;
	 char sessionID[ABM_3RD_SESSION_ID_LENGTH];
	 int nDeconChannels;
	 int nRawPSDChannels;
	 int nClassPSDChannels;
	 int nQualityCheckChannels;
	// char     chmaxNumClients; //maximum number of clients to accept
	// char     chDisconectUnrespondingClients; //whether to disconect unresponding clients and continue sending to other, or return without sending data
	// char*   pInetAddress;
	// void*   pServerptr;
}_SERVER_INFO;

typedef struct TD_TCP_DEBUG_INFO{
	int nWriterPointer;
	int nNumReaders;
	int readerPonters[ABM_3RD_MAX_NUM_READERS];
	int readerActive[ABM_3RD_MAX_NUM_READERS];
	int readerIP[ABM_3RD_MAX_NUM_READERS][4];
	int readerPort[ABM_3RD_MAX_NUM_READERS];
	int nFirstReaderIndex;
}_TCP_DEBUG_INFO;

// controls the TCP and UDP servers
THIRDPARTYCOMMUNICATION_API int  __stdcall  StartServer(_SERVER_INFO);
THIRDPARTYCOMMUNICATION_API int  __stdcall  StopServer(); /* stops both TCP and UDP */
THIRDPARTYCOMMUNICATION_API int  __stdcall  SetActiveServerPort(int nPort); 
THIRDPARTYCOMMUNICATION_API int  __stdcall  StopServerTCP();
THIRDPARTYCOMMUNICATION_API int  __stdcall  StopServerUDP();
THIRDPARTYCOMMUNICATION_API int  __stdcall  GetStatusServerTCP(unsigned int &nConnectedClients);
THIRDPARTYCOMMUNICATION_API int  __stdcall  GetStatusServerUDP(unsigned int &nReceivingClients);
THIRDPARTYCOMMUNICATION_API int  __stdcall	GetDebugInfoTCP(_TCP_DEBUG_INFO*);
THIRDPARTYCOMMUNICATION_API float*  __stdcall  GetNotifications(int &nCount);


// sends timestamps to datastreaming 
THIRDPARTYCOMMUNICATION_API int __stdcall SendDatagram(char* pData,int nLength,char* messageCode);
// sends timestamps to datastreaming via UDP
THIRDPARTYCOMMUNICATION_API int __stdcall SendUDPDatastreamingTS(unsigned int epoch, unsigned int offset, unsigned int hour,unsigned int minute,unsigned int second, unsigned int millisecond);

// sends message via TCP 
THIRDPARTYCOMMUNICATION_API int __stdcall SendDatagramTCP(char* pData,int nLength,char messageCode);

// datastreaming TCP send functions
THIRDPARTYCOMMUNICATION_API int __stdcall SendRawData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendDeconData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendPSDData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendPSDRawData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendThirdPartyData(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendEBData(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendEXCData(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendEMGData(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendSATData(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendSPKData(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendEKGData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendBrainState(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendQuality(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendQualityChannel(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendImpedance(char* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendAccelerometer(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendAngles(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendMovement(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendTCPTimeStamp(unsigned int epoch, unsigned int offset, unsigned int hour,unsigned int minute,unsigned int second, unsigned int millisecond);
THIRDPARTYCOMMUNICATION_API int __stdcall SendZScore(float* pData,int nLength); 
THIRDPARTYCOMMUNICATION_API int __stdcall SendBandOverallPSDData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendBandOverallPSDRawData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendPSDBandwidthData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendPSDBandwidthRawData(float* pData,int nLength); 
THIRDPARTYCOMMUNICATION_API int __stdcall SendPulseRateData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendHapticData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendRawOpticalData(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendNotification(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendBatteryAndImpedances(int nBatteryLevel, int nEpoch, int nImpOnlineStatus, int nImpOnlineValues[24]);
THIRDPARTYCOMMUNICATION_API int __stdcall SendMissedBlocks(int nMissedBlocks,int nEpoch);
THIRDPARTYCOMMUNICATION_API int __stdcall SetEEGChannelMapInfo(char* chMapInfoBytes,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendCurrentQuality(float* pData,int nLength);
THIRDPARTYCOMMUNICATION_API int __stdcall SendClassQuality(float* pData,int nLength);


// teaming mode functions
THIRDPARTYCOMMUNICATION_API int __stdcall EnableTeamingMode(int nReceivePort,int nSendPort); //beacon callback, port 
THIRDPARTYCOMMUNICATION_API int __stdcall StartReadingBeacons(void (__stdcall *pFunc)(int));
THIRDPARTYCOMMUNICATION_API int __stdcall SendClassificationToServer(char* pClassification,int nBeaconCounter,int nSessionNumber);
// teaming mode end

#define ABM_3RD_TM_SEND_MULTIPLE_SESSIONS 100

// Error Codes
#define     ABM_3RD_ERROR_INIT_PARAMETER    -101
#define     ABM_3RD_ERROR_INIT_WINSOCK2     -102
#define     ABM_3RD_ERROR_CREATE_SOCKET     -103
#define     ABM_3RD_ERROR_GET_HOST          -104
#define     ABM_3RD_ERROR_CONNECT           -105
#define     ABM_3RD_ERROR_NON_BLOCK_MODE    -106
#define		ABM_3RD_CAN_NOT_LISTEN_ON_PORT  -100
#define		ABM_3RD_ACCEPT_FAILED           -200

#define ABM_3RD_OPERATION_PERFORMED_SUCCESSFULLY		0 
#define ABM_3RD_ERROR_SERVER_BUFFER_OVERFLOW			-1
#define ABM_3RD_ERROR_SERVER_NOT_STARTED				-2
#define ABM_3RD_ERROR_SERVER_STATE_ERROR				-3
#define ABM_3RD_ERROR_OPERATION_NOT_SUPPORTED			-4
#define ABM_3RD_ERROR_SERVER_ALREADY_STARTED			-5
#define ABM_3RD_ERROR_SERVER_COULD_NOT_SEND				-6
#define ABM_3RD_ERROR_READERS_LIMIT_REACHED				-7
#define ABM_3RD_ERROR_ACCEPT_FAILED						-8
#define ABM_3RD_ERROR_NULL_POINTER_OPERAND				-9
#define ABM_3RD_ERROR_NO_DATA_TO_SEND					-10
#define ABM_3RD_ERROR_MESSAGE_TOO_LARGE					-11
#define ABM_3RD_ERROR_COULDNT_STOP_SERVER				-12 

#define ABM_3RD_NO_ERROR_FOUND							0

// states - TCP server
#define ABM_3RD_SERVER_STATE_TCP_NOT_STARTED			0
#define ABM_3RD_SERVER_STATE_TCP_STARTED				1
#define ABM_3RD_SERVER_STATE_TCP_STOPPED				2
#define ABM_3RD_SERVER_STATE_TCP_SERVER_ERROR			3

#define ABM_3RD_TCP_READER_STOPPED						1
#define ABM_3RD_TCP_LISTENER_STOPPED					2
#define ABM_3RD_TCP_SOCKET_ERROR						3
#define ABM_3RD_TCP_CLIENT_CLOSED_CONNECTION			4
#define ABM_3RD_TCP_SHUTDOWN_FAILED						5

// states - UDP server
#define ABM_3RD_SERVER_STATE_UDP_NOT_STARTED			0
#define ABM_3RD_SERVER_STATE_UDP_STARTED				1
#define ABM_3RD_SERVER_STATE_UDP_STOPPED				2
#define ABM_3RD_SERVER_STATE_UDP_SERVER_ERROR			3

// Data Types
#define ABM_DATA_TYPE_RAW 1
#define ABM_DATA_TYPE_DECON 2
#define ABM_DATA_TYPE_PSD 3
#define ABM_DATA_TYPE_PSD_RAW 4
#define ABM_DATA_TYPE_BAND_ALL_PSD 5
#define ABM_DATA_TYPE_BAND_ALL_PSD_RAW 6
#define ABM_DATA_TYPE_BAND_PSD 7
#define ABM_DATA_TYPE_BAND_PSD_RAW 8
#define ABM_DATA_TYPE_EKG 9
#define ABM_DATA_TYPE_CLASS 10 
#define ABM_DATA_TYPE_QUALITY 11
#define ABM_DATA_TYPE_QUALITY_CH 12
#define ABM_DATA_TYPE_TILT 13
#define ABM_DATA_TYPE_MOVEMENT 14
#define ABM_DATA_TYPE_HAPTIC 15
#define ABM_DATA_TYPE_OPTICAL_RAW 16
#define ABM_DATA_TYPE_PULSE 17
#define ABM_DATA_TYPE_ZSCORE 18
#define ABM_DATA_TYPE_ART_EB 19
#define ABM_DATA_TYPE_ART_SPK 20
#define ABM_DATA_TYPE_ART_EXC 21
#define ABM_DATA_TYPE_ART_SAT 22
#define ABM_DATA_TYPE_ART_EMG 23
#define ABM_DATA_TYPE_TIMESTAMP 24
#define ABM_DATA_TYPE_ANGLES 25
 
// datastreaming protocols
#define ABM_3RD_DATAGRAMPROTOCOL_UNDEFINED				001	
#define ABM_3RD_DATAGRAMPROTOCOL_RAW					101 // sends raw
#define ABM_3RD_DATAGRAMPROTOCOL_RAW_DECON				102 // sends raw+decon
#define ABM_3RD_DATAGRAMPROTOCOL_RAW_DECON_CLASS		103 // sends raw+decon+class
#define ABM_3RD_DATAGRAMPROTOCOL_RAW_DECON_CLASS_WL	    104 // sends raw+decon+class+workload

#define ABM_3RD_NOTIFICATION_MESSAGE_SIZE 12
#define ABM_3RD_NOTIFICATION_BUFFER_SIZE 100

//#define ABM_START_ACQUISITION_ALL 100
//#define ABM_START_ACQUISITION_SINGLE 101 // broadcast with session id to identify
//#define ABM_PING_ALL 120
//#define ABM_PING_SINGLE 121 // broadcast with session id to identify
//#define ABM_STOP_ACQUISITION_ALL 110
//#define ABM_STOP_ACQUISITION_SINGLE 110
//#define ABM_SET_DATA_PORT 130


#endif __THIRDPARTYCOMMUNICATION__

