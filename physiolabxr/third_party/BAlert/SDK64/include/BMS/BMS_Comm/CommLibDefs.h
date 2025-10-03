#pragma once

#ifndef _COMMLIBDEFS_H
#define _COMMLIBDEFS_H

#include <tchar.h>
#include <WinDef.h>

//---------------------------------------------------------------------------

#define ABM_MAX_DAU_HARDWARE_CHANNELS				24

//---------------------------------------------------------------------------

/*!
\brief CommLib error codes
*/
typedef enum
{
	ABM_COMMLIB_ERR_NO_ERROR						= 0x00000000,
	ABM_COMMLIB_ERR_APPLICATION_ERR_BASE			= 0x10000000, /* Check SetLastError MSDN reference for more info */
	ABM_COMMLIB_ERR_NOT_IMPLEMENTED					= 0x10000010,
	ABM_COMMLIB_ERR_NOT_SUPPORTED					= 0x10000020,

	ABM_COMMLIB_ERR_INVALID_PARAMETER				= 0x10000100,

	ABM_COMMLIB_ERR_INVALID_PORTINFO				= 0x10000110,
	ABM_COMMLIB_ERR_INVALID_CONNECTION_HANDLE		= 0x10000120,

	ABM_COMMLIB_ERR_INVALID_ESU_HANDLE				= 0x10000140,

	ABM_COMMLIB_ERR_INVALID_DAU_HANDLE				= 0x10000140,

	ABM_COMMLIB_ERR_NOT_SUPPORTED_BY_PROTOCOL		= 0x10000420,
	ABM_COMMLIB_ERR_INVALID_DEVICE_STATE			= 0x10000430,
	ABM_COMMLIB_ERR_INVALID_COMMAND_PARAMETER		= 0x10000440,
	ABM_COMMLIB_ERR_COMMAND_RESPONSE_TIMEOUT		= 0x10000450
} TABMCommLibError; /* SetLastError/GetLastError */

//---------------------------------------------------------------------------

/*!
\brief Supported port types
*/
typedef enum
{
	ABM_PORT_TYPE_UNKNOWN							= 0x00,
	ABM_PORT_TYPE_SERIAL							= 0x01,
	ABM_PORT_TYPE_USBEXPRESS						= 0x02
} TABMPortType;

/*!
\brief Information about port used for connecting to ABM devices
*/
typedef struct
{
	WCHAR FriendlyName[MAX_PATH];
	WCHAR DeviceInstanceID[MAX_PATH];
	TABMPortType PortType;

} TABMPortInfo;

/*!
\brief Extended information about port used for communication with ABM devices (internal use only)
*/
typedef struct
{
	TABMPortInfo PortInfo;
	WCHAR DevicePath[MAX_PATH];
} TABMPortInfoEx;

/*!
\brief Port statistics (internal use only)
*/
typedef struct
{
	DWORD TotalBytesRX;
	DWORD TotalBytesTX;
} TABMPortStat;

//---------------------------------------------------------------------------

/*!
\brief Information about connection to port used for communication with ABM devices
*/
typedef struct
{
	DWORD Handle;
	TABMPortInfo PortInfo;
} TABMConnectionInfo;

//---------------------------------------------------------------------------

/*!
\brief Information about BlueTooth device detected by ESU during searching mode
*/
typedef struct
{
	WCHAR BDA[13];
	WCHAR FriendlyName[33];
} TABMBTDeviceInfo;

//---------------------------------------------------------------------------

/*!
\brief Supported external synchronization unit types
*/
typedef enum
{
	ABM_ESU_TYPE_UNKNOWN							= 0x00,
	ABM_ESU_TYPE_DONGLE								= 0x01,
	ABM_ESU_TYPE_SINGLE_CHANNEL						= 0x02,
	ABM_ESU_TYPE_MULTI_CHANNEL						= 0x03,
	ABM_ESU_TYPE_SMART_DONGLE						= 0x04,
	ABM_ESU_TYPE_SMART_MULTI_CHANNEL				= 0x05,
	ABM_ESU_TYPE_BL654_LE_DONGLE					= 0x06
} TABMESUType;

#define IsESUSingleChannel(X) ((X == ABM_ESU_TYPE_DONGLE) || (X == ABM_ESU_TYPE_SMART_DONGLE) || (X == ABM_ESU_TYPE_SINGLE_CHANNEL) || (X == ABM_ESU_TYPE_BL654_LE_DONGLE))
#define IsADongle(X) ((X == ABM_ESU_TYPE_DONGLE) || (X == ABM_ESU_TYPE_SMART_DONGLE) || (X == ABM_ESU_TYPE_BL654_LE_DONGLE))

/*!
\brief Possible states of external synchronization unit
*/
typedef enum
{
	ABM_ESU_STATE_UNKNOWN							= 0x00,
	ABM_ESU_STATE_NOT_INITIALIZED					= 0x01,
	ABM_ESU_STATE_READY								= 0x02,
	ABM_ESU_STATE_ENUMERATING_BT_DEVICES			= 0x03,
	ABM_ESU_STATE_MEMORY_EDITING					= 0x04
} TABMESUState;

/*!
\brief ESU Bluetooth port settings
*/
typedef struct
{
	BYTE BDA[13];
	BYTE ModuleType;
	BYTE BoardType;
	BYTE Delay;
	bool TimestampProcessing;
} TABMESUBTPortSettings;

/*!
\brief ESU serial port settings
*/
typedef struct
{
	BYTE Protocol;
	BYTE Delay;
} TABMESUSERPortSettings;

/*!
\brief ESU parallel port settings
*/
typedef struct
{
	BYTE Protocol;
	BYTE Delay;
} TABMESUPARPortSettings;

/*!
\brief Information about external synchronization unit
*/
typedef struct
{
	DWORD Handle;
	TABMConnectionInfo ConnectionInfo;
	bool Initialized;
	BYTE ProtocolVersion;

	TABMESUType ESUType;

	WCHAR FirmwareVersion[5];
	WCHAR SerialNumber[10];

	WCHAR FriendlyName[33];

	TABMESUBTPortSettings BTPortSettings[2];
	TABMESUSERPortSettings SERPortSettings[4];
	TABMESUPARPortSettings PARPortSettings;

	BYTE Debug;
} TABMESUInfo;

//---------------------------------------------------------------------------

/*!
\brief Supported data acquisition unit types
*/
typedef enum
{
	ABM_DAU_TYPE_UNKNOWN							= 0x00,
	ABM_DAU_TYPE_06CH_16BIT							= 0x01,
	ABM_DAU_TYPE_04CH_16BIT_TILT_IRED				= 0x01,
	ABM_DAU_TYPE_06CH_24BIT							= 0x02,
	ABM_DAU_TYPE_09CH_24BIT							= 0x03,
	ABM_DAU_TYPE_04CH_24BIT							= 0x04,
	ABM_DAU_TYPE_10CH_16BIT_TILT					= 0x05,
	ABM_DAU_TYPE_04CH_16BIT_TILT					= 0x06,
	ABM_DAU_TYPE_10CH_16BIT							= 0x07,
	ABM_DAU_TYPE_10CH_24BIT							= 0x08,
	ABM_DAU_TYPE_10CH_24BIT_TILT					= 0x09,
	ABM_DAU_TYPE_24CH_16BIT							= 0x0A,
	ABM_DAU_TYPE_24CH_16BIT_TILT					= 0x0B
} TABMDAUType;

/*!
\brief Possible states of data acquisition unit
*/
typedef enum
{
	ABM_DAU_STATE_UNKNOWN							= 0x00,
	ABM_DAU_STATE_NOT_INITIALIZED					= 0x01,
	ABM_DAU_STATE_READY								= 0x02,
	ABM_DAU_STATE_MEMORY_EDITING					= 0x03,
	ABM_DAU_STATE_ACQUISITION						= 0x04,
	ABM_DAU_STATE_IMPEDANCE							= 0x05,
	ABM_DAU_STATE_TECHNICAL_MONITORING				= 0x06
} TABMDAUState;

/*!
\brief Possible working submodes of data acquisition unit
*/
typedef enum
{
	ABM_WORKING_SUBMODE_UNKNOWN						= 0x0000,
	ABM_WORKING_SUBMODE_STARTUP_INIT				= 0x0111,
	ABM_WORKING_SUBMODE_STARTUP_BTINIT				= 0x0112,
	ABM_WORKING_SUBMODE_STARTUP_BTHOSTINIT			= 0x0113,
	ABM_WORKING_SUBMODE_STARTUP_BTHOSTWAIT			= 0x0114,
	ABM_WORKING_SUBMODE_STARTUP_BTSLAVEINIT			= 0x0115,
	ABM_WORKING_SUBMODE_STARTUP_BTSLAVEWAIT			= 0x0116,
	ABM_WORKING_SUBMODE_BOOTSTRAP_INIT				= 0x0221,
	ABM_WORKING_SUBMODE_BOOTSTRAP_SDUPLOAD			= 0x0222,
	ABM_WORKING_SUBMODE_BOOTSTRAP_BTUPLOAD			= 0x0223,
	ABM_WORKING_SUBMODE_INIT_MANAGEMENT				= 0x0331,
	ABM_WORKING_SUBMODE_INIT_STUDY					= 0x0332,

	ABM_WORKING_SUBMODE_ACQUISITION_BT_SD			= 0x0441,
	ABM_WORKING_SUBMODE_ACQUISITION_BT				= 0x0442,
	ABM_WORKING_SUBMODE_ACQUISITION_SD				= 0x0443,
	ABM_WORKING_SUBMODE_ACQUISITION_SHUTDOWN		= 0x0444,
	ABM_WORKING_SUBMODE_ACQUISITION_ERROR			= 0x0445,
	ABM_WORKING_SUBMODE_ACQUISITION_TIMEOUT			= 0x0446,
	ABM_WORKING_SUBMODE_ACQUISITION_VOLTAGE			= 0x0447,
	ABM_WORKING_SUBMODE_COMMUNICATION_ACTIVE		= 0x0551,
	ABM_WORKING_SUBMODE_ERROR_HW					= 0x0EE1,
	ABM_WORKING_SUBMODE_ERROR_BATTLOW				= 0x0EE2,
	ABM_WORKING_SUBMODE_ERROR_BATTERR				= 0x0EE3,
	ABM_WORKING_SUBMODE_LOWPOWER_OFF				= 0x0FF1,
	ABM_WORKING_SUBMODE_LOWPOWER_TIMEOUT			= 0x0FF2
} TABMDAUWorkingSubmode;

/* !
\brief Reconfigurable channel gains settings of data acquisition unit
*/
//typedef BYTE TABMReconfigurableChannelGains[4];

/*!
\brief Gain calibration settings of data acquisition unit
*/
typedef short TABMGainCalibrationData[ABM_MAX_DAU_HARDWARE_CHANNELS];

/*!
\brief Impedance calibration settings of data acquisition unit
*/
typedef short TABMImpedanceCalibrationData[ABM_MAX_DAU_HARDWARE_CHANNELS];

/*!
\brief Channels pin mapping settings of data acquisition unit
*/
typedef BYTE TABMChannelsPinMappingData[2 * ABM_MAX_DAU_HARDWARE_CHANNELS];

/*!
\brief DC offset settings of data acquisition unit
*/
typedef short TABMEEGDCOffsetData[ABM_MAX_DAU_HARDWARE_CHANNELS];

/*!
\brief Information about data acquisition unit
*/
typedef struct
{
	DWORD Handle;
	TABMConnectionInfo ConnectionInfo;
	bool Initialized;
	BYTE ProtocolVersion;

	WCHAR SerialNumberHost[17];
	WCHAR HardwareVersionHost[5];
	WCHAR FirmwareVersionHost[12];

	WCHAR SerialNumberHostSys[17];
	WCHAR ReleaseHostSys[9];

	WCHAR SerialNumberAcq[17];
	WCHAR HardwareVersionAcq[5];
	WCHAR FirmwareVersionAcq[12];

	WCHAR SerialNumberAcqSys[17];
	WCHAR ReleaseAcqSys[9];

	WCHAR SerialNumberAdc[17];
	WCHAR HardwareVersionAdc[5];

	short TxOffset;
	short RxOffset;
	bool TiltTransmitted;
	bool CalibrationMemoryPresent;
	bool CalibrationMemoryInitialized;
	WCHAR BDA[13];

	WORD HeadsetImpedancePinConfiguration;

	/* 4x16, 4x24, 9x16, 9x24, 10x16, 10x24, 24x16 */
	WORD SamplingRate; /* 128, 256, 512, ... */
	BYTE NumberOfChannels; /* 4, 9, 10, 24 */
	bool IRedChannelPresent;
	bool ExtendedX4Packet;
	bool MICChannelPresent;
	bool IRedRawChannelPresent;
	bool TCChannelsPresent;
	BYTE SampleSizeInBytes; /* 2, 3 */
	BYTE SampleBlocksPerPacket; /* 1 for 24ch, 2 for all other */
	BYTE SampleBlockSizeInBytes; /* NumberOfChannels * SampleSizeInBytes */
	BYTE AcquisitionDataSizeInBytes; /* SampleBlocksPerPacket * SampleBlockSizeInBytes */
	BYTE AccelerometerDataSizeInBytes; /* 0, 6 */
	BYTE AcquisitionPacketSizeInBytes; /* 5 + AcquisitionDataSizeInBytes + AccelerometerDataSizeInBytes */
	bool ECGChannelPresent;	
	BYTE ECGChannelNumber;

	WCHAR FriendlyName[33];

	TABMGainCalibrationData GainCalibrationData;
	TABMImpedanceCalibrationData ImpedanceCalibrationData;
	TABMChannelsPinMappingData ChannelsPinMappingData;
	TABMEEGDCOffsetData EEGDCOffsetData;

	float GainCoefficients[24];
} TABMDAUInfo;

//---------------------------------------------------------------------------

/*!
\brief Information about data acquisition unit
*/
typedef struct
{
	DWORD DAUHandle;

	BYTE BlockType;
	BYTE ByteOrder;

	BYTE ProtocolStatus; // ?

	BYTE Unused01;

	WORD WorkingSubMode;

	WORD SamplingRate; /* 128, 256, 512, ... */

	BYTE NumberOfChannels; /* 4, 9, 10, 24 */
	bool IRedChannelPresent;
	bool RedChannelPresent;
	bool MicChannelPresent;
	bool ECGChannelPresent;
	BYTE ECGChannelNumber;

	BYTE SampleSizeInBytes; /* 2, 3 */
	BYTE SampleBlocksPerPacket; /* 1 for 4ch and 24ch, 2 for all other */
	BYTE SampleBlockSizeInBytes; /* NumberOfChannels * SampleSizeInBytes */
	BYTE AcquisitionDataSizeInBytes; /* SampleBlocksPerPacket * SampleBlockSizeInBytes */
	BYTE AccelerometerDataSizeInBytes; /* 0, 6 */
	BYTE AcquisitionPacketSizeInBytes; /* 5 + AcquisitionDataSizeInBytes + AccelerometerDataSizeInBytes */

//#pragma message (__LOCATION__ "TABMAcquisitionDataInfo: Define this structure!!!\n")
} TABMAcquisitionDataInfo;

/*!
\brief One 16-bit raw sample for one channel
*/
typedef struct
{
	BYTE Low;
	BYTE High;
} TABMSample16;

/*!\brief One 24-bit raw sample for one channel
*/
typedef struct
{
	BYTE High;
	BYTE Mid;
	BYTE Low;
} TABMSample24;

/*!
\brief Set of raw samples for all channels
*/
typedef struct
{
	WORD Counter;
	union
	{
		BYTE Bytes[72]; /* Size of this array is a size of largest array in bytes */
		TABMSample16 Samples16[ABM_MAX_DAU_HARDWARE_CHANNELS];
		TABMSample24 Samples24[ABM_MAX_DAU_HARDWARE_CHANNELS];
	};
} TABMRawSampleInfo; /* NOTE - Holds bytes for ONE sample only */

/*!
\brief Set of integer samples for all channels (internal use only)
*/
typedef struct
{
	WORD Counter;
	int Samples[ABM_MAX_DAU_HARDWARE_CHANNELS];
} TABMSampleIntegersInfo;

/*!
\brief Set of float samples for all channels
*/
typedef struct
{
	WORD Counter;
	float Samples[ABM_MAX_DAU_HARDWARE_CHANNELS];
} TABMSampleInfo;

/*!
\brief Accelerometer data
*/
typedef struct
{
	WORD Counter;
	short X;
	short Y;
	short Z;
} TABMTiltInfo;

/*!
\brief Optical channel data
*/
typedef struct
{
	WORD Counter;
	WORD IRed;
} TABMIRedInfo;

/*!
\brief Optical channel data
*/
typedef struct
{
	WORD Counter;
	WORD Red;
} TABMRedInfo;

/*!
\brief Mic data
*/
typedef struct
{
	WORD Counter;
	WORD Mic;
} TABMMicInfo;

/*!
\brief Data acquired from slow channels
*/
typedef struct
{
	WORD PassCounter;
	WORD BatteryVoltageWORD;
	float BatteryVoltage;
	WORD VCCWORD;
	float VCC;
	WORD TemperatureHostWORD;
	float TemperatureHost;
	WORD TemperatureAcqWORD;
	float TemperatureAcq;
	BYTE ImpedanceLevelAndChannel;
	WORD RfSerialNumber;
	DWORD Timestamp;
	BYTE CustomMarkA;
	BYTE CustomMarkB;
	BYTE CustomMarkC;
	BYTE CustomMarkD;
	BYTE OnLineImpStatus;
	BYTE OnLineImpValues[24];
	WORD TechEvent;
} TABMSlowChannelsInfo;

/*!
\brief Timestamp acquired from external synchronization unit
*/
typedef struct
{
//#ifdef LONG_COUNTER
	int Counter;
//#else
//	WORD Counter;
//#endif
	DWORD Timestamp;
} TABMTimestampInfo;

/*!
\brief Miscellaneous data acquired
*/
typedef struct
{
	WORD Counter;
	WORD Haptic;
} TABMProbeDataInfo;


/*!
\brief IRedRAw data acquired
*/
typedef struct
{
	WORD Counter;
	WORD IRedRaw;
} TABMIRedRawInfo;


/*!
\brief Technical channel data acquired
*/
typedef struct
{
	WORD Counter;
	BYTE BatteryVoltage;
    BYTE ImpedanceR1R2;
    BYTE ImpedanceR3R4;
    BYTE AlarmAlert;
} TABMTechnicalChannelInfo;


//---------------------------------------------------------------------------

/*!
\brief Pointer to application-defined callback function used for enumerating supported ports available on system
*/
typedef bool (__stdcall *PABMEnumPortsProc)(const TABMPortInfoEx*, void*, bool bReadDeviceDetails, bool bReadESUDetails);

/*!
\brief Pointer to application-defined callback function used for enumerating established connections
*/
typedef bool (__stdcall *PABMEnumConnectionsProc)(const TABMConnectionInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for enumerating BlueTooth devices in range
*/
typedef bool (__stdcall *PABMEnumBTDevicesProc)(const TABMBTDeviceInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for enumerating available external synchronization units
*/
typedef bool (__stdcall *PABMEnumESUsProc)(const TABMESUInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for enumerating available data acquisition units
*/
typedef bool (__stdcall *PABMEnumDAUsProc)(const TABMDAUInfo*, void*);

//---------------------------------------------------------------------------

/*!
\brief Pointer to application-defined callback function used for unbuffered output of raw samples
*/
typedef void (__stdcall *PABMOnNewRawSample)(const TABMRawSampleInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of samples
*/
typedef void (__stdcall *PABMOnNewSample)(const TABMSampleInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of accelerometer data
*/
typedef void (__stdcall *PABMOnNewTilt)(const TABMTiltInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of optical channel data
*/
typedef void (__stdcall *PABMOnNewIRed)(const TABMIRedInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of optical channel data
*/
typedef void (__stdcall *PABMOnNewRed)(const TABMRedInfo*, void*);


/*!
\brief Pointer to application-defined callback function used for real time notification about new data acquired from slow channels
*/
typedef void (__stdcall *PABMOnNewSlowChannels)(const TABMSlowChannelsInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of timestamps
*/
typedef void (__stdcall *PABMOnNewTimestamp)(const TABMTimestampInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of third party data
*/
typedef void (__stdcall *PABMOnNewThirdPartyData)(const BYTE*, const DWORD, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of probe data
*/
typedef void (__stdcall *PABMOnNewProbeData)(const TABMProbeDataInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of MIC data
*/
typedef void (__stdcall *PABMOnNewMicData)(const TABMMicInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of IRedRaw data
*/
typedef void (__stdcall *PABMOnNewIRedRawData)(const TABMIRedRawInfo*, void*);

/*!
\brief Pointer to application-defined callback function used for unbuffered output of Technical Channel data
*/
typedef void (__stdcall *PABMOnNewTechnicalChannel)(const TABMTechnicalChannelInfo*, void*);



//---------------------------------------------------------------------------


#endif
