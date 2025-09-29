#pragma once

// consts
#define BALERT_MAX_NUM_CHANNELS 24
#define BALERT_MAX_CH_NAME 20
#define BALERT_MAX_EVENT_DATA_SIZE 1024
#define ABM_DATA_X4_CHANNELS 4
#define ABM_DATA_X24_CHANNELS 24
#define ABM_DATA_X10_CHANNELS 10
#define MAX_NUM_ELECTRODE	 24
#define	MAX_LENGTH_CHANNEL_NAME		20
#define	MAX_NUM_EEGCHANNELS				24

typedef struct TD_ELECTRODES_INFO_BMS{	
	int     nNumElectrodes;
	int     nStabilization;
	int     nAgregationsSamples;
	int     nCurrentType;
	char    cElName[MAX_NUM_ELECTRODE][MAX_LENGTH_CHANNEL_NAME];//[in]the name of electrode max 20 characters
	int     nElCommand[MAX_NUM_ELECTRODE];// Impedance command to be sent for this electrode to be measured
	int     nElChannel[MAX_NUM_ELECTRODE];// EEG channel to be used when measuring electrode
	int     nElReferentialElectrode[MAX_NUM_ELECTRODE];// Electrode to be used when substracting ref el. (-1 for none)
}_ELECTRODES_INFO_BMS;   

typedef struct TD_EEGCHANNELS_INFO_BMS{
	char    cChName[MAX_NUM_EEGCHANNELS][MAX_LENGTH_CHANNEL_NAME];//[in]the name of channels max 20 characters
	BOOL    bChUsed[MAX_NUM_EEGCHANNELS];// whether channel is used or not		
	BOOL    bChUsedInQualityData[MAX_NUM_EEGCHANNELS];// whether channel is used or not for quality
	BOOL	bChCanBeDecontaminated[MAX_NUM_EEGCHANNELS];
	BOOL	bIsChEEG[MAX_NUM_EEGCHANNELS];
	BOOL	nFirstElectrode[MAX_NUM_EEGCHANNELS];
	BOOL	nSecondElectrode[MAX_NUM_EEGCHANNELS];
}_EEGCHANNELS_INFO_BMS;   

typedef struct TD_AUXDATA_INFO_BMS{	
	BOOL	bIred;
	BOOL	bRed;
	BOOL	bTilt;
	int 	nEcgIndex;	
	BOOL	bMic;
	BOOL	bHaptic;
}_AUXDATA_INFO_BMS;   

typedef struct TD_HARDWARE_INFO_BMS{	
	int nBatteryMax; //millivolts
	int nBatteryMin; //millivolts	
	int nTiltLinearTransformA;
	int nTiltLinearTransformB;
}_HARDWARE_INFO_BMS;   


typedef struct TD_SESSIONTYPES_INFO_BMS{	
	BOOL	bDecon; //whether decontamination is supported or not
	BOOL	bBalert;//whether b-alert classification is supported or not
	BOOL	bWorkload;	//whether workload calculation is supported or not
}_SESSIONTYPES_INFO_BMS;   

typedef struct TD_CHANNELMAP_INFO_BMS{
	int nDeviceTypeCode;
	int nSize;
	_EEGCHANNELS_INFO_BMS stEEGChannels;
	_ELECTRODES_INFO_BMS stElectrodes;	
	_AUXDATA_INFO_BMS stAuxData;
	_HARDWARE_INFO_BMS stHardwareInfo;
	_SESSIONTYPES_INFO_BMS stSessionTypes;	
}_CHANNELMAP_INFO_BMS;  

typedef struct TD_BMS_SETTINGS{
	bool bNewImpedanceMeasuring;
	bool bCreateMBStat;
}_BMS_SETTINGS;  

typedef struct __TD_DEVICE_INFO_BMS{
	int  nConnectionStatus; //port opened
	int  nNumberOfChannels; //number of channel
	char chChannelMap[BALERT_MAX_NUM_CHANNELS*BALERT_MAX_CH_NAME]; // Ch(N)-ChName - comma separated
	int  nBatteryPercentage; 	
	char chDeviceName[256];//device's name
	int nESU_TYPE;	
	int nHEADSET_TYPE; //old 1, flex 8 - 2, flex x24t 3
}_BALERT_DEVICE_INFO_BMS;

typedef struct _DATA_HEADER_BMS{		
	float Counter; 
	float ESUtimestamp;
	float hour;
	float minute;
	float second;
	float millisecond;
	float packetSize;
}_BALERT_DATA_HEADER_BMS;
 
typedef struct _EVENT_HEADER_BMS{		
	float Counter; 
	float EEGSampleCounter;
	float ESUtimestamp;
	float hour;
	float minute;
	float second;
	float millisecond;
	float packetSize;	
}_BALERT_EVENT_HEADER_BMS;

typedef struct TD_DATA_X4_DATA_BMS{
	_BALERT_DATA_HEADER_BMS header; 
	float samples[ABM_DATA_X4_CHANNELS];
	float tiltX;
	float tiltY;
	float tiltZ;
}_BALERT_DATA_X4_PACKET_BMS;

typedef struct TD_DATA_X10_DATA{
	_BALERT_DATA_HEADER_BMS header; 
	float samples[ABM_DATA_X10_CHANNELS];
	float tiltX;
	float tiltY;
	float tiltZ;
}_BALERT_DATA_X10_PACKET;

typedef struct TD_DATA_X24_DATA{
	_BALERT_DATA_HEADER_BMS header; 
	float samples[ABM_DATA_X24_CHANNELS];
	float tiltX;
	float tiltY;
	float tiltZ;
}_BALERT_DATA_X24_PACKET;

typedef struct TD_DATA_MIC_DATA{
	_BALERT_DATA_HEADER_BMS header; 
	float micValue;		
}_BALERT_DATA_MIC_PACKET;

typedef struct TD_DATA_RAW_OPTICAL_DATA{
	_BALERT_DATA_HEADER_BMS header; 
	float opticalRawValue;		
}_BALERT_DATA_RAW_OPTICAL_PACKET;

typedef struct TD_DATA_OPTICAL_DATA{
	_BALERT_DATA_HEADER_BMS header; 
	float opticalValue;		
}_BALERT_DATA_OPTICAL_PACKET;


typedef struct TD_DATA_EVENT{
	_BALERT_EVENT_HEADER_BMS header; 		
	char data[BALERT_MAX_EVENT_DATA_SIZE];
}_BALERT_DATA_EVENT;

typedef struct TD_IMPEDANCE_RESULT{
	int  channelNumber;
	char channelName[BALERT_MAX_CH_NAME];	
	int   firstElectrodeImpedanceStatus; 
	char  firstElectrodeName[BALERT_MAX_CH_NAME]; 
	float firstElectrodeValue; 
	int   secondElectrodeImpedanceStatus; 
	char  secondElectrodeName[BALERT_MAX_CH_NAME];	
	float secondElectrodeValue; 	
}_BALERT_IMPEDANCE_RESULT;

typedef struct TD_BMS_RETURN_VALUE{
	static const int BMS_API_COMMAND_SUCCESS = 1;
	static const int BMS_API_COMMAND_FAILED = 0;
	static const int IMP_COULDNT_START = 0;

static const int X4_DATA_SIZE_BMS = 1;
	static const int DEV_INFO_OK = 1;
	static const int DEV_INFO_NO_DEVICE = 0;
	static const int DEV_INFO_WRONG_DEVICE_TYPE = -1;
	static const int DEV_INFO_WRONG_USB_CONFIGURATION = -2;
	static const int DEV_INFO_MISSING_CHANNEL_MAP_CONFIGURATION = -3;
	static const int BMS_INVALID_FLEX_INFO = -4;
	static const int BMS_START_ACQUISITION_FAILED = -5;
	static const int BMS_INVALID_SEQUENCE = -6;
}_BALERT_CONSTANTS_RETURN_VALUES;

typedef struct TD_BMS_DEVICE{
	static const int MAX_NUM_CH_BMS = 24;
	static const int MAX_NUMBER_SLOW_CHANNEL_BMS = 64;
    static const int EPOCH_SZ_BMS = 256;
	static const int SIZE_PACKET_COUNTER_BMS = 64; 		
	static const int SAMPLE_RATE_BMS	=   256;
	static const int X4_DATA_SIZE_BMS	=   (4+3);
	static const int X10_DATA_SIZE	=   (10+3);
	static const int X24_DATA_SIZE	=   (24+3);
	static const int EVENT_TYPE_API = 0;
	static const int EVENT_TYPE_ESU = 1;
}_BALERT_CONSTANTS_DEVICE_BMS;

#define _BALERT_CONSTANTS_DEVICE_SAMPLES_DELAY 3.90625

typedef struct TD_BMS_BUFFERS{
	static const int MAX_PACKET_SIZE = 34;	
	static const int MAX_EVENTS_NUM  = 1000;
	static const int SAMPLE_BUFFER_SIZE = 25600;
}_BALERT_CONSTANTS_BUFFERS;

typedef struct TD_BMS_OPERATING_MODE{
	static const int  BMS_IMPEDANCE_MODE = 1;
	static const int  BMS_DATA_MODE_WAIT_DATA = 1;
	static const int  BMS_DATA_MODE_GET_DATA = 2;
	static const int  BMS_DATA_MODE_UNINITIALIZED = 0;
}_BALERT_CONSTANTS_OPERATING_MODE;

#define BALERT_MAX_NUM_CHANNELS 24
#define BALERT_MAX_CH_NAME 20
#define BALERT_MAX_EVENT_DATA_SIZE 256
#define ABM_DATA_X4_CHANNELS 4
#define ABM_DATA_X24_CHANNELS 24
#define ABM_DATA_X10_CHANNELS 10
#define MAX_NUM_ELECTRODE	 24
#define	MAX_LENGTH_CHANNEL_NAME		20
#define	MAX_NUM_EEGCHANNELS				24
#define TILT_SIZE 3

#define		ABM_INVALID_VALUE				-1	

//device type code in ini device channel map
#define		X10CODE			10
#define		X24CODE			24
#define		X8CODE_MID		101
#define		X8CODE_REF		102
#define		X8CODE_SP		103
#define		X8CODE_SPP		104

//strip codes and macros
#define STRIP_10_20				0x2021
#define STRIP_10_20_REDUCED		0x2123
#define STRIP_10_20_LM			0x2232
#define STRIP_10_20_REDUCED_LM	0x2239
#define STRIP_SENSOR_HARNESS    0x2230
#define Is10CHStrip(x) (x==STRIP_10_20_REDUCED || x==STRIP_10_20_REDUCED_LM)
#define IsStripLESupported(x) (x==STRIP_10_20_LM || x==STRIP_10_20_REDUCED_LM|| x==STRIP_SENSOR_HARNESS)
#define IsSupportedStrip(x) (x==STRIP_10_20 || x==STRIP_10_20_REDUCED || x==STRIP_10_20_LM || x==STRIP_10_20_REDUCED_LM || x==STRIP_SENSOR_HARNESS)

//device config functionalities error codes
#define BMS_DEVCONFIG_SUCCESS			0
#define	BMS_DEVCONFIG_NO_ESU_FOUND		-0x1
#define BMS_DEVCONFIG_DEVICE_NOT_FOUND	-0x2
#define BMS_DEVCONFIG_ALREADY_CONNECTED	-0x3
#define BMS_DEVCONFIG_CONNECTION_FAILED	-0x4
#define BMS_DEVCONFIG_NOT_X_SERIES		-0x5
#define BMS_DEVCONFIG_NOT_OLD_RECIEVER	-0x6
#define BMS_DEVCONFIG_SEARCH_FAILED		-0x7
#define BMS_DEVCONFIG_SYNC_FAILED		-0x8
#define BMS_DEVCONFIG_UPLOAD_FW_FAILED  -0x9
#define BMS_DEVCONFIG_PARSE_FW_FAILED	-0xA
#define BMS_DEVCONFIG_RESET_FAILED		-0xB
#define BMS_DEVCONFIG_BOOT_FAILED		-0xC
#define BMS_DEVCONFIG_READ_MC_FAILED	-0xD
#define BMS_DEVCONFIG_WRITE_MC_FAILED	-0xE
#define BMS_DEVCONFIG_CHECK_MC_FAILED	-0xF
#define BMS_DEVCONFIG_ESU_WIRED			-0x10
#define BMS_DEVCONFIG_ESU_NOT_CONFIGURED			-0x11

#define BMS_OBSOLETE_NO_LONGER_IN_USE    -0xFF
