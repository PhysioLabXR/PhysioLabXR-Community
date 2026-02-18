#ifndef  __TDEXPORT_DEFINE__
#define  __TDEXPORT_DEFINE__

#define ABM_DATA_SESSION_ID_LENGTH					 9
#define ABM_DATA_NUM_NOTIFICATION_PARAMS	        10
#define ABM_DATA_NUM_COMMAND_PARAMS					4	
#define ABM_DATA_MAX_NUM_CHANNELS					24 
#define	ABM_DATA_PSD_BINS							128
#define ABM_DATA_SIZE_BW							1024
#define ABM_DATA_SIZE_BW_OA							60
#define ABM_DATA_SIZE_ZSCORE						606


enum _ABM_DATA_NOTIFICATION_TYPE 
{
	ABM_DATA_CLIENT_COMMAND = 110, 
	ABM_DATA_SERVER_RESPONSE = 111, 
	ABM_TEAMING_COMMAND = 120,
	ABM_TEAMING_RESPONSE = 121,
    ABM_3RD_DATASTREAMING_INFO = 200,
	ABM_3RD_BEACON = 201,
};


enum _ABM_DATA_COMMAND_SYB_TYPE 
{
	ABM_DATA_COMMAND_START_ACQUISITION = 301,
	ABM_DATA_COMMAND_STOP_ACQUISITION = 302,
	ABM_DATA_COMMAND_PAUSE_ACQUISITION = 303,
	ABM_DATA_COMMAND_RESUME_ACQUISITION = 304,
	ABM_DATA_COMMAND_CHECK_IMPEDANCE = 305,
	ABM_DATA_COMMAND_CHECK_DATAQUALITY = 306,
	ABM_DATA_COMMAND_GET_SDK_MODE = 307,
	ABM_DATA_COMMAND_INIT_SESSION = 308,
	ABM_DATA_COMMAND_CHECK_SELECTED_IMPEDANCE = 309,
};

enum _ABM_DATA_DS_INFO_SYB_TYPE 
{
	ABM_3RD_DS_INFO_BATTERY = 100,
    ABM_3RD_DS_INFO_MISSED_BLOCKS = 101,	
	ABM_3RD_DS_INFO_ERROR = 200,
	ABM_3RD_DS_INFO_ELECTRODE_FINISHED = 210,
};


typedef struct __ATIME_DATA_WITH_OFFSET{
	float epoch;
	float offset;
	float abmHour;
	float abmMin;
	float abmSec;
	float abmMilli;	
}_ABM_DATA_TIME;

typedef struct __ATIME_DATA_NO_OFFSET{
	float epoch;	
	float abmHour;
	float abmMin;
	float abmSec;
	float abmMilli;	
}_ABM_DATA_TIME_NO_OFFSET;

typedef struct __QUALITY_DATA{
	_ABM_DATA_TIME time; 
	float practice;
	float training;
	float testing;
}_ABM_DATA_QUALITY_OVERALL;

typedef struct __AEKG_DATA{
	_ABM_DATA_TIME time; 
	float heartRate;
	float interBeatInterval;
	float beatQuality;
	float packetType;
}_ABM_DATA_EKG;

//typedef struct __APULSE_RATE_DATA{
//	_ABM_DATA_TIME time; 
//	float heartRate;
//	float interBeatInterval;
//	float beatQuality;
//	float packetType;
//}_ABM_DATA_PULSE_RATE;

typedef struct __MOVEMENT_DATA{
	_ABM_DATA_TIME_NO_OFFSET time; 
	float value;
	float level;	
}_ABM_DATA_MOVEMENT;

//typedef struct __HAPTIC_DATA{
//	_ABM_DATA_TIME time; 
//	float state;	
//}_ABM_DATA_HAPTIC;

typedef struct __PSD_BANDWIDTH_DATA{	
	_ABM_DATA_TIME_NO_OFFSET time; 
	float bandwidth[ABM_DATA_SIZE_BW];	
}_ABM_DATA_BANDWIDTH;

typedef struct __PSD_BANDWIDTH_OVERALL_DATA{	
	_ABM_DATA_TIME_NO_OFFSET time; 
	float bandwidthOverall[ABM_DATA_SIZE_BW_OA];	
}_ABM_DATA_BANDWIDTH_OVERALL;

typedef struct __ZSCORE_DATA{	
	_ABM_DATA_TIME_NO_OFFSET time; 
	float zscore[ABM_DATA_SIZE_ZSCORE];	
}_ABM_DATA_ZSCORE;

typedef struct __RAW_OPTICAL_DATA{
	_ABM_DATA_TIME time; 
	float sampleValue;	
}_ABM_DATA_RAW_OPTICAL;

typedef struct __AACC_DATA{
	_ABM_DATA_TIME time; 
	float rawTiltX;
	float rawTiltY;
	float rawTiltZ;
}_ABM_DATA_ACCELEROMETER;

typedef struct __AAngles_DATA{
	_ABM_DATA_TIME time; 
	float angleX;
	float angleY;
	float angleZ;
}_ABM_DATA_ANGLES;


typedef struct __TD_DEVICE_INFO{
	wchar_t  chDeviceName[256];//device's name
	int   nCommPort; //comm port
	int   nECGPos; //ecg position
	int   nNumberOfChannel; //number of channel
	int   nNumberPSDBands; //number of PSD bandwidths
	int   nNumberPSDBandsOverall; //number of PSD bandwidths (overall)
	int  nDeconChannels;
	int  nPSDRawChannels;
	int  nPSDClassChannels;
	int  nQualityCheckChannels;
	char  sessionID[ABM_DATA_SESSION_ID_LENGTH]; //SessionID
}_ABM_DATA_DEVICE_INFO;

typedef struct __TD_BRAIN_STATE{
	_ABM_DATA_TIME_NO_OFFSET time; 
	float 	fClassificationEstimate; //information about classification
	float 	fHighEngagementEstimate; //information about high-engagement
	float 	fLowEngagementEstimate; //information about low-engagement
	float 	fDistractionEstimate; //information about distraction
	float 	fDrowsyEstimate; //information about drowsy
	float 	fWorkloadFBDSEstimate; //information about workload - task FBDS
	float 	fWorkloadBDSEstimate; //information about workload - task BDS
	float 	fWorkloadAverageEstimate; //information about workload - average across tasks
}_ABM_DATA_BRAIN_STATE;

typedef struct __TD__NOTIFICATION{
	float   fNotificationType;
	float   fNotificationSubType;
	float   fNotificationParams[ABM_DATA_NUM_NOTIFICATION_PARAMS];
}_ABM_DATA_NOTIFICATION;
 
typedef struct __TD_COMMAND_INFO_DATA{
// notification used for commands and responses
	float   fNotificationType;
	float   fNotificationSubType;
	float	sequenceNumber;
    float	timestampHour;
    float	timestampMinute;
    float	timestampSecond;
    float	timestampMillisecond;
    float	commandStatus;
	float   fCommandParams[ABM_DATA_NUM_COMMAND_PARAMS];
}_ABM_DATA_COMMAND_INFO;

typedef struct TD_SAMPLES_DATA{
	_ABM_DATA_TIME time; 
	float samples[ABM_DATA_MAX_NUM_CHANNELS];
}_ABM_DATA_SAMPLES;

typedef struct TD_QUALITY_CHANNEL_DATA{
	_ABM_DATA_TIME time; 
	float values[2*ABM_DATA_MAX_NUM_CHANNELS];
}_ABM_DATA_QUALITY_PER_CHANNEL;


typedef struct TD_BINS_DATA{
	_ABM_DATA_TIME_NO_OFFSET time; 
	float bins[ABM_DATA_MAX_NUM_CHANNELS][ABM_DATA_PSD_BINS];
}_ABM_DATA_BINS;

typedef struct __TD_ELECTRODE{
	char  chName[20]; //name of electrode
	BOOL  bImpedance; //if impedance is well (low)
	float fImpedanceValue; //value of impedance
}_ABM_DATA_ELECTRODE;

typedef struct TD_DATA_CLASSIFICATION_PACKAGE 
{
	int nBeaconValue; 
	int sessionInfo;	
    _ABM_DATA_BRAIN_STATE brainState;
}_ABM_DATA_CLASSIFICATION_PACKAGE; 


#endif //__TDEXPORT_DEFINE__