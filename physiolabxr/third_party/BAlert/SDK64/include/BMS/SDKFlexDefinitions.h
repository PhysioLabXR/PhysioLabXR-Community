#pragma once

#include <atlstr.h>
#define  MAX_NUM_CHANNELS 24
#define  MAX_NUM_ELECTRODE 24

struct DEVICE_INFO
{
public:
	DEVICE_INFO()
	{
		memset(chDeviceName,0,256 * sizeof(char));
		memset(ucSerial1, 0, 17 * sizeof(char));

		memset(ucBtSerialNum, 0, 13 * sizeof(char));
		memset(ucFirmwareVersion, 0, 12 * sizeof(char));
		memset(ucHardwareVersion, 0, 5 * sizeof(char));
		memset(ucESUFirmwareVersion, 0, 5 * sizeof(char));

		memset(chDeviceInstanceID, 0, MAX_PATH * sizeof(char));

		nCommPort = -1;
		nECGPos = -1;
		nNumberOfChannel = -1;
		nBytesPerSample = -1;
		bTilt = FALSE;
		bHasESU = FALSE;
		bMultiChESU = FALSE;
	}

	char  chDeviceName[256];//device's name
	int   nCommPort; //comm port
	int   nECGPos; //ecg position
	int   nNumberOfChannel; //number of channel
	int   nESUType; //type of connecting device
	int   nTymestampType; // type of timestamp
	int   nDeviceHandle; // handle to identify device   
	char chDeviceInstanceID[MAX_PATH];
	int	 nBytesPerSample;
	BOOL	 bTilt;
	BOOL     bHasESU;
	BOOL	 bMultiChESU;
	char ucSerial1[17];

	char ucBtSerialNum[13];
	char ucFirmwareVersion[12];
	char ucHardwareVersion[5];
	char ucESUFirmwareVersion[5];

	int  nChannelsConfigurationAdc[MAX_NUM_CHANNELS * 2];
	int  nDCoffsetAdc[MAX_NUM_CHANNELS];

	int  nImpedanceOffsetAdc[MAX_NUM_CHANNELS];
	int  nImpedanceConstAdc[MAX_NUM_ELECTRODE];
};

struct RECEIVER_INFO
{
	RECEIVER_INFO()
	{
		nHeadsetTypeOnBT1 = -1;
		nHeadsetTypeOnBT2 = -1;
		nBTModule1Type = -1;
		nBTModule2Type = -1;
		nThirdPartyDataTypeSerial1 = -1;
		nThirdPartyDataTypeSerial2 = -1;
		nThirdPartyDataTypeSerial3 = -1;
		nThirdPartyDataTypeSerial4 = -1;
		nThirdPartyDataTypeParalel = -1;
		fHeadset1Delay = 0;
		fHeadset2Delay = 0;
		fSerialPort1Delay = 0;
		fSerialPort2Delay = 0;
		fSerialPort3Delay = 0;
		fSerialPort4Delay = 0;
		fParalelPortDelay = 0;
		bProcessTimeStamp = FALSE;
		bMultiChEsu = FALSE;

		memset(ucESUSerialNum,0,10 * sizeof(unsigned char));
		memset(ucDebugMask, 0, 9 * sizeof(unsigned char));
		memset(ucBTSerialNum1, 0, 13 * sizeof(unsigned char));
		memset(ucBTSerialNum2, 0, 13 * sizeof(unsigned char));
		memset(ucESUFirmwareVersion, 0, 5 * sizeof(unsigned char));
	}

	int nHeadsetTypeOnBT1;
	int nHeadsetTypeOnBT2;
	int nBTModule1Type;
	int nBTModule2Type;
	int nThirdPartyDataTypeSerial1;
	int nThirdPartyDataTypeSerial2;
	int nThirdPartyDataTypeSerial3;
	int nThirdPartyDataTypeSerial4;
	int nThirdPartyDataTypeParalel;
	float fHeadset1Delay;
	float fHeadset2Delay;
	float fSerialPort1Delay;
	float fSerialPort2Delay;
	float fSerialPort3Delay;
	float fSerialPort4Delay;
	float fParalelPortDelay;
	BOOL	bProcessTimeStamp;
	unsigned char ucESUSerialNum[10];
	unsigned char ucBTSerialNum1[13];
	unsigned char ucBTSerialNum2[13];
	unsigned char ucDebugMask[9];
	unsigned char ucESUFirmwareVersion[5];
	unsigned char ucESUName[30];
	BOOL	bMultiChEsu;
};





struct FLEX_DEVICE_INFO
{
	FLEX_DEVICE_INFO()
	{
			nXDeviceName = 0;
			nXDeviceModel = 0;		
			nSensorConfiguration = 0;
			strDeviceSN = "";
			strFirmwareVersion = "";
			strHardwareRevision= "";
			nACQType = 0;		
			nRecordingMode = 0;
			nAuxChannel  = 0;	
			nAudioOn = 0;
	}
	int				nXDeviceName;			
	int				nXDeviceModel;		
	int				nSensorConfiguration;	
	CString			strDeviceSN;			
	CString			strFirmwareVersion;	
	CString			strHardwareRevision;	
	int				nACQType;				
	int				nRecordingMode;		
	int				nAuxChannel;
	int				nAudioOn;
};

typedef struct TD_SDK_EXPORTED_PARAMS
{
	BOOL bCreateSyncFile;
	BOOL bCopyRawFile;
	BOOL bCopySystemDatFile;
	BOOL bConvertWholeRaw;
	WCHAR tchSDCardDownloadPath[MAX_PATH];
}_SDK_EXPORTED_PARAMS;

typedef struct _CBStudyInfo 
{
	SYSTEMTIME		stStartTime;
	SYSTEMTIME		stEndTime;
	int				nSessionIndexStart;
	int				nSessionIndexEnd;
	int				nDuration;  // in minutes
	WCHAR			strGuid[MAX_PATH];
	bool 			bInclude;
	bool			bDownloaded;
} CBStudyInfo;

