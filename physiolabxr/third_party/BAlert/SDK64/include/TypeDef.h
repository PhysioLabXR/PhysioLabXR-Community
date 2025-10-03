
#ifndef  __TD_DEFINE__
#define  __TD_DEFINE__


#define   __ESU_TIME_STAMP_LENGTH	4
#define   __SYS_TIME_LENGTH			8
#define   ABM_MAX_CONNECTED_DEVICES 10

#include "EegAcqDefine.h"

#pragma pack (1)


typedef struct TD_ELECTRODE{
   wchar_t  chName[20]; //name of electrode
   BOOL  bImpedance; //if impedance is well (low)
   float fImpedanceValue; //value of impedance
}ELECTRODE;

typedef struct TD_CHANNEL_INFO{
   wchar_t  chName[20]; //name of electrode
   BOOL  bTechnicalInfo; //if impedance is well (low)
}CHANNEL_INFO;

typedef struct TD_DEVICE_INFO{
   wchar_t  chDeviceName[256];//device's name
   int   nCommPort; // comm port 
   int   nECGPos; //ecg position
   int   nNumberOfChannel; //number of channel
   int   nESUType; //type of connecting device
   int   nTymestampType; // type of timestamp
   int   nDeviceHandle; // handle to identify device   
   wchar_t  chDeviceID[MAX_PATH]; // ESU/dongle port device ID
}_DEVICE_INFO;

typedef struct TD_STATUS_INFO{
	float BatteryVoltage;
	int BatteryPercentage;
	//int Temperature;
	int Timestamp;
	int TotalMissedBlocks;   
	int ABMSDK_Mode;
	int LastErrorCode;
	int CustomMarkA;
	int CustomMarkB;
	int CustomMarkC;
	int CustomMarkD;
	int nTotalSamplesReceived;
	int OnLineImpStatus;
	int OnLineImpValues[24];
	int TechEvent;
}_STATUS_INFO;



//CWPC
typedef struct TD_BRAIN_STATE{
   float    fEpoch;
   float    fABMSDKTimeStampHour;//time stamp
   float    fABMSDKTimeStampMinute;//time stamp
   float    fABMSDKTimeStampSecond;//time stamp
   float    fABMSDKTimeStampMilsecond;//time stamp
   float 	fClassificationEstimate; //information about classification
   float 	fHighEngagementEstimate; //information about high-engagement
   float 	fLowEngagementEstimate; //information about low-engagement
   float 	fDistractionEstimate; //information about distraction
   float 	fSleepOnsetEstimate; //information about drowsy
   float 	fWorkloadFBDS; //information about FBDS workload
   float 	fWorkloadBDS; //information about BDS workload
   float 	fWorkloadAverage; //information about workload average
}_BRAIN_STATE;

typedef struct TD_SESSION_INFO{
	wchar_t  	chDefinitionPath[1024]; //full path of definition file (only needed for GetBrainState)
	wchar_t  	chDestionationPath[1024]; //full path for destination(ebs) file 
	DWORD    wRawChannels; //markers for which channel raw data need to be provided;// LSB used for first channel...if the corresponding bit is 1, raw data will be provided for that channel
	DWORD    wDecChannels; //markers for which channel dec data need to be provided;// LSB used for first channel...if the corresponding bit is 1, raw data will be provided for that channel
	DWORD    wPsdChannels; //markers for which channel PSD need to be calculated;// LSB used for first channel...if the corresponding bit is 1, PSD will be calculated for that channel
	DWORD    dwPSD[MAX_NUM_EEGCHANNELS][4];//markers for 128 bins for one channel. LSB is marker for 1Hz bin…
	char     chName[MAX_NUM_EEGCHANNELS][MAX_LENGTH_CHANNEL_NAME];//[in]the name of channels max 20 characters
	int     inputPinPositive[MAX_NUM_EEGCHANNELS];// holds positive pin asignment for each channel
	int     inputPinNegative[MAX_NUM_EEGCHANNELS];// holds negative pin asignment for each channel (not used for referential)
	int     channelImpedanceType[MAX_NUM_EEGCHANNELS];// IMPEDANCE_REFERENTIAL 0, IMPEDANCE_DIFERENTIAL 1, IMPEDANCE_NOT_AVAILABLE	2
	BOOL	bApply65HzFilter; // not used (should be removed)
	BYTE      bBrainState; // markers for (classification, high-engagement, low-engagement, distraction, drowsy, workload
	BYTE      bPlayEbsMode; // play ebs mode	
}_SESSION_INFO;   

typedef struct TD_ELECTRODES_INFO{	
	int     nNumElectrodes;
	int     nStabilization;
	int     nAgregationsSamples;
	int     nCurrentType;
	wchar_t    cElName[MAX_NUM_ELECTRODE][MAX_LENGTH_CHANNEL_NAME];//[in]the name of electrode max 20 characters
	int     nElCommand[MAX_NUM_ELECTRODE];// Impedance command to be sent for this electrode to be measured
	int     nElChannel[MAX_NUM_ELECTRODE];// EEG channel to be used when measuring electrode
	int     nElReferentialElectrode[MAX_NUM_ELECTRODE];// Electrode to be used when substracting ref el. (-1 for none)
}_ELECTRODES_INFO;   

typedef struct TD_EEGCHANNELS_INFO{
	char    cChName[MAX_NUM_EEGCHANNELS][MAX_LENGTH_CHANNEL_NAME];//[in]the name of channels max 20 characters
	BOOL    bChUsed[MAX_NUM_EEGCHANNELS];// whether channel is used or not		
	BOOL    bChUsedInQualityData[MAX_NUM_EEGCHANNELS];// whether channel is used or not for quality
	BOOL	bChCanBeDecontaminated[MAX_NUM_EEGCHANNELS];
	BOOL	bIsChEEG[MAX_NUM_EEGCHANNELS];
	BOOL	bIsFlex;
	int		nChannelMap;
}_EEGCHANNELS_INFO;   

typedef struct TD_AUXDATA_INFO{	
	BOOL	bIred;
	BOOL	bRed;
	BOOL	bTilt;
	int 	nEcgIndex;	
	BOOL	bMic;
	BOOL	bHaptic;
}_AUXDATA_INFO;   

typedef struct TD_HARDWARE_INFO{	
	int nBatteryMax; //millivolts
	int nBatteryMin; //millivolts	
	int nTiltLinearTransformA;
	int nTiltLinearTransformB;
}_HARDWARE_INFO;   


typedef struct TD_SESSIONTYPES_INFO{	
	BOOL	bDecon; //whether decontamination is supported or not
	BOOL	bBalert;//whether b-alert classification is supported or not
	BOOL	bWorkload;	//whether workload calculation is supported or not
}_SESSIONTYPES_INFO;   

typedef struct TD_CHANNELMAP_INFO{
	int nDeviceTypeCode;
	int nSize;
	_EEGCHANNELS_INFO stEEGChannels;
	_ELECTRODES_INFO stElectrodes;	
	_AUXDATA_INFO stAuxData;
	_HARDWARE_INFO stHardwareInfo;
	_SESSIONTYPES_INFO stSessionTypes;	
}_CHANNELMAP_INFO;   


typedef struct TD_SESSION_INFO_PE{
	int nNumberOfChannels;
	int nEKG;
	BOOL	bApply65HzFilter;
	wchar_t chName[MAX_NUM_EEGCHANNELS][MAX_LENGTH_CHANNEL_NAME];//[in]the name of channels max 20 characters
}_SESSION_INFO_PE;

typedef union __ESU_TIME_STAMP{
   float  time_ms;
   BYTE   pByteStream[__ESU_TIME_STAMP_LENGTH];
}_ESU_TIME_STAMP;

typedef struct __EXTERN_CALC_TABLE_NAME{
	int     iRealIndex;		//index into block
	int     iIndex;         //index into calculated block
	int     iCalcIndex1;	//real index from real block of data
	int     iCalcIndex2;	//real index from real block of data
}_EXTERN_CALC_TABLE_NAME;

// REFCOM moved struct to typedef.h
typedef struct __BALERT_DATA_INFO{
	int                  nPlayedSamples;	// samples already in channels
	int                  nNotReceiving;		// elapsed time in milliseconds when nothing is received from device
	int                  nTotalSeconds;		// total number of seconds in EBS file
	int                  nFirstSecond;		// first second in stream and buffers
	int                  nActiveSecond;		// active second = slider position for EBS file
	void*	      pActivTable;	// pointer to active table
	int                  nActiveScreen;//which screen is presented	
} _BALERT_DATA_INFO;

typedef struct __ELECTRODE{
	wchar_t    elName[50];
	int     nHirose;//hirose plug-in
	int     nSwitch;//switch which need to press
	int     nChannel;//where that electrode will be seen on screen
	int     nSubChannel;//what channel we need to substract to get correct values 
	int     nSubEl;//what electrode we need to substract to get correct values 
	float   fImpedance;
	int     nNumberOfSaturation;//how many times saturation found on this electrode
}_ELECTRODE;

typedef struct __ESU_PORT_INFO{
	int   nESU_TYPE;	 // UNKNOWN - 0, DONGLE - 1,SINGLE_CH 2,MULTI_CH - 3
	int   bWired; // Wired - 1, Wireless - 0
	int   nSerialPortType[ABM_THIRD_PARTY_PORTS_NUM];
	int   nParalelPortType;
	int   bIsRegularConfig; // Regular - 1, Irregular - 0
	wchar_t BDA[13];
} _ESU_PORT_INFO;

typedef struct __BALERT_FIRMWARE_VERSION{
	wchar_t ucFirmwareVersion[12];
	wchar_t ucMCESUFirmwareVersion[5];
	wchar_t ucDONGLEFirmwareVersion[5];
	} _BALERT_FIRMWARE_VERSION;

//typedef struct __ESU_PORT_INFO_DETAILS{
//	int   nESU_TYPE;	 // UNKNOWN - 0, DONGLE - 1,SINGLE_CH 2,MULTI_CH - 3
//	int   bWired; // Wired - 1, Wireless - 0
//	int   nSerialPortType[ABM_THIRD_PARTY_PORTS_NUM];
//	int   nParalelPortType;
//	int   bIsRegularConfig; // Regular - 1, Irregular - 0
//	char BDA[13];
//	char FirmwareVersion[5];
//	char SerialNumber[10];
//
//} _ESU_PORT_INFO_DETAILS;


struct _DEVICE_INFO_DETAILS
{
public:
	_DEVICE_INFO_DETAILS()
	{
		memset(chDeviceName,0,256 * sizeof(wchar_t));
		memset(ucSerialReceiver, 0, 17 * sizeof(wchar_t));
		memset(ucSerialHeadset, 0, 17 * sizeof(wchar_t));

		memset(ucBtSerialNum, 0, 13 * sizeof(wchar_t));
		memset(ucFirmwareVersion, 0, 12 * sizeof(wchar_t));
		memset(ucHardwareVersion, 0, 5 * sizeof(wchar_t));
		memset(ucESUFirmwareVersion, 0, 5 * sizeof(wchar_t));

		memset(chDeviceInstanceID, 0, MAX_PATH * sizeof(wchar_t));

		nCommPort = -1;
		nECGPos = -1;
		nNumberOfChannel = -1;
		nBytesPerSample = -1;
		nESUType = -1;
		bTilt = FALSE;
		
	}

	wchar_t  chDeviceName[256];//device's name
	int   nCommPort; //comm port
	int   nECGPos; //ecg position
	int   nNumberOfChannel; //number of channel
	int   nESUType; //type of connecting device
    int   nTymestampType; // type of timestamp
    int   nDeviceHandle; // handle to identify device   
	wchar_t chDeviceInstanceID[MAX_PATH];
	int	 nBytesPerSample;
	BOOL	 bTilt;

	wchar_t ucSerialHeadset[17];
	wchar_t ucSerialReceiver[17];

	wchar_t ucBtSerialNum[13];
	wchar_t ucFirmwareVersion[12];
	wchar_t ucHardwareVersion[5];
	wchar_t ucESUFirmwareVersion[5];

	int  nChannelsConfigurationAdc[MAX_NUM_EEGCHANNELS * 2];
	int  nDCoffsetAdc[MAX_NUM_EEGCHANNELS];

	int  nImpedanceOffsetAdc[MAX_NUM_EEGCHANNELS];
	int  nImpedanceConstAdc[MAX_NUM_EEGCHANNELS];

	


};

 struct _LSLSTREAMING_INFO {

public:
	_LSLSTREAMING_INFO()
	{
		bLSLRawEEG = false;
		bLSLFilteredEEG = false;
		bLSLTimestamp = false;
		bLSLBAlertMetric = false;
		bLSLHR = false;
		bLSLHRV = false;
		bLSLHRVFreq = false;


	}
	BOOL bLSLRawEEG;
	BOOL bLSLFilteredEEG;
	BOOL bLSLTimestamp;
	BOOL bLSLBAlertMetric;
	BOOL bLSLHR;
	BOOL bLSLHRV;
	BOOL bLSLHRVFreq;


	void operator =(_LSLSTREAMING_INFO& info)
	{
		bLSLBAlertMetric = info.bLSLBAlertMetric;
		bLSLFilteredEEG = info.bLSLFilteredEEG;
		bLSLRawEEG = info.bLSLRawEEG;
		bLSLTimestamp = info.bLSLTimestamp;
		bLSLHR = info.bLSLHR;
		bLSLHRV = info.bLSLHRV;
		bLSLHRVFreq = info.bLSLHRVFreq;

	}

	bool StreamingRequested()
	{
		return bLSLBAlertMetric || bLSLFilteredEEG || bLSLRawEEG || bLSLTimestamp || bLSLHR || bLSLHRV || bLSLHRVFreq;
	}


	
};

#pragma pack()


#endif //__TD_DEFINE__
