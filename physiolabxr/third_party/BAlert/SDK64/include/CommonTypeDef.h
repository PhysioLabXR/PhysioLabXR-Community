
#include <stdio.h>
#include <tchar.h>

#pragma once

#define		BUFFER_SIZE_IN_RECORDS		100
#define     MAX_EDF_NUM_ALLCHANNELS		100 //the number of channels including time channels (X24 has 24 eeg + 2 time ch)
#define		CHANNEL_NAME_LENGTH			20
#define		MAX_STR_TMP                 2048
#define		MAXSIZE_ERROR_MSG           1000

#define		FILE_CHANNELTYPES_Raw		0
#define		FILE_CHANNELTYPES_Decon		1
#define		FILE_CHANNELTYPES_RawDecon	2
#define		FILE_CHANNELTYPES_Events	3

#define		FILE_STORAGETYPE_Ebs			0
#define		FILE_STORAGETYPE_Edf			1
#define		FILE_STORAGETYPE_EdfPSignal		2
#define		FILE_STORAGETYPE_EdfPEvent		3
#define		FILE_STORAGETYPE_EdfFromSDCard	4

#define		X24_QEEG			0
#define		X24_STANDARD		1
#define		X10_STANDARD		2
#define		X4_B_ALERT			3
#define		X4_APPT				4
#define		X4_SLEEP_PROFILER	5
#define		X24t_10_20			6
#define		X10t_STANDARD		7
#define		X24t_REDUCED		8
#define		X24t_10_20_LM		9
#define		X24LE_10_20_LM		10
#define		X24t_10_20_LM_Red	11
#define		X24LE_10_20_LM_Red	12
#define		X10t_10_20_LM_Red	13
#define		X24LE_Ambulatory	14

#define DEFAULT_VALUE	-99999



struct _BALERT_LAB_HEADER
	{
	int		nFileStorageType;		// file type (one of constants FILE_STORAGETYPE_... )
	int		nSubjectNo;				//label for subject that is performing acquisition
	int		nSessionType;			//type of session being performed
	int		nGroup;					//group of current session
	int		nStudyType;				//current study type
	int		nIteration;				//current session iteration
	int		nDeviceType;			//type of device used for acquisition
	char	chDeviceType[20];
	char	chFimrwareVersion[12];
	char	chESUFimrware[10];
	char	chHardwareVersion[10];
	char	chSoftwareVersion[20];
	int		nChannelsNmb;
	TCHAR	chChannelName[MAX_EDF_NUM_ALLCHANNELS][CHANNEL_NAME_LENGTH];
	int		nChannelSampleRate[MAX_EDF_NUM_ALLCHANNELS];
	int		nChannelType[MAX_EDF_NUM_ALLCHANNELS];
	int		nChannelByteSize[MAX_EDF_NUM_ALLCHANNELS];
	float	fChannelGain[MAX_EDF_NUM_ALLCHANNELS];
	float	fChannelDCOffset[MAX_EDF_NUM_ALLCHANNELS];
	char	chChannelFilters[MAX_EDF_NUM_ALLCHANNELS];
	char	chRecordingTime[20];
	char	chSerialNum[30];

	char	chSortDesc[50];
	char	chFullDesc[100];
	char	chHardwareSoftwareID[MAX_STR_TMP];
	int		nSamplesNmb[MAX_EDF_NUM_ALLCHANNELS];



	int		nESUTimestampIndex;
	int 	nSysTimestampIndex;
	int		nEventsIndex;
	int		nECGChannelIndex;


	// EDF specific 
	char	chReserved[44];
	int		nNumOfDataRecords; // number of epochs in EDF
	int		nEDFHeaderSize;
	char	chTransducerType[MAX_EDF_NUM_ALLCHANNELS][80];
	char	chPhysicalDimension[MAX_EDF_NUM_ALLCHANNELS][8];
	float	fPhysicalMinimum[MAX_EDF_NUM_ALLCHANNELS];
	float	fPhysicalMaximum[MAX_EDF_NUM_ALLCHANNELS];
	int		nDigitalMinimum[MAX_EDF_NUM_ALLCHANNELS];
	int		nDigitalMaximum[MAX_EDF_NUM_ALLCHANNELS];
	char	chPrefiltering[MAX_EDF_NUM_ALLCHANNELS][80];
	char	chUserID[40];
	char	chESUSerial[12];
	};


struct _BALERT_LAB_FILEINFO
	{
	int nSubjectNo;				//label for subject that is performing acquisition
	int	nSessionType;			//type of session being performed
	int	nGroup;					//group of current session
	int nStudyType;				//current study type
	int nIteration;				//current session iteration



	int		nChannelsNmb;
	TCHAR	chChannelName[MAX_EDF_NUM_ALLCHANNELS][CHANNEL_NAME_LENGTH];
	int		nChannelSampleRate[MAX_EDF_NUM_ALLCHANNELS];
	int		nChannelType[MAX_EDF_NUM_ALLCHANNELS];
	int		nChannelByteSize[MAX_EDF_NUM_ALLCHANNELS];
	float	fChannelGain[MAX_EDF_NUM_ALLCHANNELS];

	TCHAR	chRecordingTime[20];
	TCHAR	chSoftwareVersion[20];
	int		nSamplesNmb[MAX_EDF_NUM_ALLCHANNELS];
	int		nFileEegChannelTypes;

	int		nESUTimestampIndex;
	int 	nSysTimestampIndex;
	int		nEventsIndex;


	int		nESUStartTime;
	int 	nESUStopTime;
	int		nSysTimeStartH;
	int		nSysTimeStartMin;
	int		nSysTimeStartSec;
	int		nSysTimeStartMSec;

	int		nSysTimeStopH;
	int		nSysTimeStopMin;
	int		nSysTimeStopSec;
	int		nSysTimeStopMSec;
	int		nFileStorageType;		// file type (one of constants FILE_STORAGETYPE_... )

	float	fChannelDCOffset[MAX_EDF_NUM_ALLCHANNELS];
	TCHAR	chFimrwareVersion[12];  //12 instead of 10, keep alignment as factor of 4 to marshal it easier
	TCHAR	chESUFimrware[12];
	TCHAR	chUserID[40];
	TCHAR	chDeviceType[20];
	TCHAR	chHardwareVersion[12];
	TCHAR	chESUSerial[12];
	TCHAR	chDeviceSerial[32];
	};


#define PARADIGM_3RD_PARTY		0

#define PARADIGM_VPVT			1
#define PARADIGM_APVT			2
#define PARADIGM_3CVT		3
#define PARADIGM_PAL		4
#define PARADIGM_VMS		5
#define PARADIGM_VPA		6
#define PARADIGM_FDS		7
#define PARADIGM_BDS		8
#define PARADIGM_SIR		9
#define PARADIGM_IIR		10
#define PARADIGM_SNR		12
#define PARADIGM_NIR		11
#define PARADIGM_PNNL		13
#define PARADIGM_UCSD	13

#define PARADIGM_FA1B		14
#define PARADIGM_FA2B		15
#define PARADIGM_DA1B		16
#define PARADIGM_DA2B		17

#define PARADIGM_FA1BSND		18
#define PARADIGM_FA2BSND		19
#define PARADIGM_DA1BSND		20
#define PARADIGM_DA2BSND		21

#define PARADIGM_EORESTING	22
#define PARADIGM_ECRESTING	23
#define PARADIGM_AAO		35
#define PARADIGM_PAO		36

#define PARADIGM_EIR1	24
#define PARADIGM_EIR2	25
#define PARADIGM_EIR3	26

#define PARADIGM_VMS_CI		30
#define PARADIGM_3CVT_CI		27
//#define PARADIGM_SIR_CI		28
//#define PARADIGM_SIR_ALT_CI		29
#define PARADIGM_EIR3_PART1    28
#define PARADIGM_EIR3_PART2    29

#define PARADIGM_ANY		30

#define PARADIGM_EIR1_PART1		37
#define PARADIGM_EIR1_PART2		38
#define PARADIGM_EIR1_PART3		39
#define PARADIGM_EIR2_PART1		40
#define PARADIGM_EIR2_PART2		41
#define PARADIGM_EIR2_PART3		42
#define PARADIGM_VMS1			43
#define PARADIGM_VMS2			44
#define PARADIGM_VMS3			45
#define PARADIGM_VPAO			46
#define PARADIGM_3CVT_10m		49
#define PARADIGM_VEP			51
#define PARADIGM_PASSIVE_EIR1			52
#define PARADIGM_PASSIVE_EIR2			53

#define MOBILE_AMP_PARADIGM_3CVT		1
#define MOBILE_AMP_PARADIGM_EO		2
#define MOBILE_AMP_PARADIGM_EC	3
#define MOBILE_AMP_PARADIGM_PAL	4

#define PARADIGM_SUBTYPE_VIS_FA1B		0
#define PARADIGM_SUBTYPE_VIS_FA2B		1
#define PARADIGM_SUBTYPE_VIS_DA1B		2
#define PARADIGM_SUBTYPE_VIS_DA2B		3
#define PARADIGM_SUBTYPE_AUD_FA1B		4
#define PARADIGM_SUBTYPE_AUD_FA2B		5
#define PARADIGM_SUBTYPE_AUD_DA1B		6
#define PARADIGM_SUBTYPE_AUD_DA2B		7

#define PARADIGM_UNKNOWN		-9999


#define TASK_PART_Practice				1
#define TASK_PART_PracticeTraining		2
#define TASK_PART_PracticeTesting		3
#define TASK_PART_Training				4
#define TASK_PART_Testing				5
#define TASK_PART_DS_Level				11
#define TASK_PART_PracticeTrainingA		12
#define TASK_PART_PracticeTestingA		13
#define TASK_PART_PracticeTrainingB		22
#define TASK_PART_PracticeTestingB		23
#define TASK_PART_PracticeTrainingC		32
#define TASK_PART_PracticeTestingC		33


enum enmBaselineEEGQuality
{
   enmBaselineNone,
   enmBaselineGood,
   enmBaselineMarginal,
   enmBaselineBad,

};


enum enmBaselinePerformanceQuality
{
   enmBaselinePerformanceNone,
   enmBaselinePerformanceGood,
   enmBaselineMarginalForNormative,
   enmBaselineBadForNormative,
   enmBaselinePerformanceBad
};

struct _BALERT_LAB_BASELINE_QUALITY
{
	_BALERT_LAB_BASELINE_QUALITY()
	{
		reset();

	};


	void reset()
	{
		m_fSleepOnsetPercentage = 0;
		m_fDistractedPercentage = 0;
		m_fLowEngagementPercentage = 0;
		m_fHighEngagementPercentage = 0;
		m_pEGMPercentage = 0;
		m_fInvPercentage = 0;
		m_fAverageSleepOnset = 0;
		m_fAverageDistracted = 0;
		m_fAverageLowEngagement = 0;
		m_fAverageHighEngagement = 0;
		m_nSleepOnsetEpochs = 0;
		m_nDistractedEpochs = 0;
		m_nLowEngagementEpochs = 0;
		m_nHighEngagementEPochs = 0;
		m_nEGMEpochs = 0;
		m_nINVEpochs = 0;
		m_nEMGInvEpochs = 0;
		m_nEpTotal = 0;
		m_nValidEp = 0;
		m_nTaskType = 0;
		m_fMarginalArtifactFrom = 0;
	    m_fMarginalArtifactTo = 0;
		m_fGoodClassFrom = 0;
		m_fMarginalFrom = 0;

	
		m_fGoodArtifactTo = 0;
		m_fPercentageCorrectMean = 0;
		m_fReactionTimeMean = 0;

		m_enmPredictionQuality = enmBaselineNone;
		m_enmEMGQuality = enmBaselineNone;                           
		m_enmInvQuality = enmBaselineNone;                    
		m_enmInvAndENGQuality = enmBaselineNone;
		m_enmPerformanceQuality = enmBaselinePerformanceNone;
	

		m_fPercentageCorrect = 0;
		m_fMeanRT = 0;
		m_nCorrectNmb = 0;
		m_nMissedNmb = 0;
		m_nSlowNmb = 0;
	//	m_nMissedAndSlowRTFrom = 0;
		m_fMarginalFromResponsesNmb = 0;
		m_fMarginalToResponsesNmb = 0;


		memset(m_chStatus,0,50*sizeof(char));
		memset(m_chPerformanceStatus,0,50*sizeof(char));

	}


	/////////////////////////
	void _BALERT_LAB_BASELINE_QUALITY::operator =(const _BALERT_LAB_BASELINE_QUALITY &quality)
	{
		m_fSleepOnsetPercentage = quality.m_fSleepOnsetPercentage;
		m_fDistractedPercentage = quality.m_fDistractedPercentage;
		m_fLowEngagementPercentage = quality.m_fLowEngagementPercentage;
		m_fHighEngagementPercentage = quality.m_fHighEngagementPercentage;
		m_pEGMPercentage = quality.m_pEGMPercentage;
		m_fInvPercentage = quality.m_fInvPercentage;
		m_fAverageSleepOnset = quality.m_fAverageSleepOnset;
		m_fAverageDistracted = quality.m_fAverageDistracted;
		m_fAverageLowEngagement = quality.m_fAverageLowEngagement;
		m_fAverageHighEngagement = quality.m_fAverageHighEngagement;
		m_nSleepOnsetEpochs = quality.m_nSleepOnsetEpochs;
		m_nDistractedEpochs = quality.m_nDistractedEpochs;
		m_nLowEngagementEpochs = quality.m_nLowEngagementEpochs;
		m_nHighEngagementEPochs = quality.m_nHighEngagementEPochs;
		m_nEGMEpochs = quality.m_nEGMEpochs;
		m_nINVEpochs = quality.m_nINVEpochs;
		m_nEMGInvEpochs = quality.m_nEMGInvEpochs;
		m_nEpTotal = quality.m_nEpTotal;
		m_nValidEp = quality.m_nValidEp;
		m_nTaskType = quality.m_nTaskType;
		m_fMarginalArtifactFrom = quality.m_fMarginalArtifactFrom;
	    m_fMarginalArtifactTo = quality.m_fMarginalArtifactTo;
		m_fGoodClassFrom = quality.m_fGoodClassFrom;
		m_fMarginalFrom = quality.m_fMarginalFrom;

		
		m_fGoodArtifactTo = quality.m_fGoodArtifactTo;
		m_fPercentageCorrectMean = quality.m_fPercentageCorrectMean;
		m_fReactionTimeMean = quality.m_fReactionTimeMean;

		m_enmPredictionQuality = quality.m_enmPredictionQuality;
		m_enmEMGQuality = quality.m_enmEMGQuality;                           
		m_enmInvQuality = quality.m_enmInvQuality;                    
		m_enmInvAndENGQuality = quality.m_enmInvAndENGQuality;
		m_enmPerformanceQuality = quality.m_enmPerformanceQuality;
	

		m_fPercentageCorrect = quality.m_fPercentageCorrect;
		m_fMeanRT = quality.m_fMeanRT;
		m_nCorrectNmb = quality.m_nCorrectNmb;
		m_nMissedNmb = quality.m_nMissedNmb;
		m_nSlowNmb = quality.m_nSlowNmb;
		//m_nMissedAndSlowRTFrom = quality.m_nMissedAndSlowRTFrom;
		m_fMarginalFromResponsesNmb = quality.m_fMarginalFromResponsesNmb;
		m_fMarginalToResponsesNmb = quality.m_fMarginalToResponsesNmb;



		_tcscpy(m_chStatus,quality.m_chStatus);
		_tcscpy(m_chPerformanceStatus,quality.m_chPerformanceStatus);

		
	}

	///////////////////////////////
	void _BALERT_LAB_BASELINE_QUALITY::SetType(int nClass) 
	{ 
			
		m_nTaskType = nClass;
		m_fGoodArtifactTo = 3;
		m_fMarginalArtifactFrom = 3;
		m_fMarginalArtifactTo = 10;
 
		if(nClass == PARADIGM_VPVT)
		{
			m_fGoodClassFrom = 70;
			m_fMarginalFrom = 64;


		//	m_nMissedAndSlowRTFrom = 5;
			m_fMarginalFromResponsesNmb = 90;
			m_fMarginalToResponsesNmb = 96;
		}
		else if(nClass == PARADIGM_APVT)
		{
			m_fGoodClassFrom = 82;
			m_fMarginalFrom = 76;

		
			

			
			//m_nMissedAndSlowRTFrom = 5;
			m_fMarginalFromResponsesNmb = 90;
			m_fMarginalToResponsesNmb = 96;
		}
		else if(nClass == PARADIGM_3CVT)
		{
			m_fGoodClassFrom = 75;
			m_fMarginalFrom = 67.5;

			

			
			m_fPercentageCorrectMean = 92;
			m_fReactionTimeMean = 0.745000;

		}
	}

	

	float m_fSleepOnsetPercentage;
	float m_fDistractedPercentage;
	float m_fLowEngagementPercentage;
	float m_fHighEngagementPercentage;
	float m_pEGMPercentage;
	float m_fInvPercentage;
	float m_fAverageSleepOnset;
	float m_fAverageDistracted;
	float m_fAverageLowEngagement;
	float m_fAverageHighEngagement;
	float m_fMarginalArtifactFrom;
	float m_fMarginalArtifactTo;
	float m_fGoodClassFrom;
	float m_fMarginalFrom;


	float m_fGoodArtifactTo;
	float m_fPercentageCorrectMean;
	float m_fReactionTimeMean;
	
	int	  m_nSleepOnsetEpochs;
	int   m_nDistractedEpochs;
	int	  m_nLowEngagementEpochs;
	int	  m_nHighEngagementEPochs;
	int	  m_nEGMEpochs;
	int	  m_nINVEpochs;
	int	  m_nEMGInvEpochs;
	int   m_nEpTotal;
	int	  m_nValidEp;
	int	 m_nTaskType;
	//int m_nMissedAndSlowRTFrom;
	float m_fMarginalFromResponsesNmb;
	float m_fMarginalToResponsesNmb;




	enmBaselineEEGQuality	m_enmPredictionQuality;
	enmBaselineEEGQuality m_enmEMGQuality;                           
	enmBaselineEEGQuality m_enmInvQuality;                    
	enmBaselineEEGQuality m_enmInvAndENGQuality;
	enmBaselinePerformanceQuality m_enmPerformanceQuality;

	float	m_fPercentageCorrect;
	float	m_fMeanRT;
	int		m_nCorrectNmb;
	int		m_nMissedNmb;
	int		m_nSlowNmb;



	TCHAR m_chStatus[50];
	TCHAR m_chPerformanceStatus[50];

};


struct BALERT_ARTIFACT_INFO
{
public:

	BALERT_ARTIFACT_INFO()
	{
		nArtifactType = DEFAULT_VALUE;
		nRule = DEFAULT_VALUE;
		nStartDataPoint = DEFAULT_VALUE;
		nStopDataPoint = DEFAULT_VALUE;
		memset(ucChannelName,0, 20*sizeof(char));
	}

	int nArtifactType;
	int nStartDataPoint;
	int nStopDataPoint;
	char ucChannelName[20];
	int nRule;


};

struct BALERT_OVERALL_ARTIF_INFO
{
	float allArtifactPercentage;
	float allArtifactPercentageNewEB;
	float otherArtifactsPercentage;
	float EBPercentage;
	float NewEBPercentage;

	BALERT_OVERALL_ARTIF_INFO()
	{
		allArtifactPercentage = -1;
		otherArtifactsPercentage = -1;
		EBPercentage = -1;
		allArtifactPercentageNewEB = -1;
		NewEBPercentage = -1;

	}
};

struct BALERT_CHANNEL_ARTIF_INFO
{
	BALERT_CHANNEL_ARTIF_INFO()
	{
			memset(ucChannelName,0, sizeof(char)*20);
			nZeroValuesInserted = -1;
			allArtifactPercentage = -1;
			otherArtifactsPercentage = -1;
			EBPercentage = -1;
			SatPercentage = -1;
			SpkPercentage = -1;
			ExcPercentage = -1;
			MBPercentage = -1;
			nInvalid = -1;
			area1ArtifactsPercentage = -1;
			area2ArtifactsPercentage = -1;
			area3ArtifactsPercentage = -1;
			area4ArtifactsPercentage = -1;

			allArtifactPercentageNewEB = -1;
			NewEBPercentage = -1;
			area1ArtifactsPercentageNewEB = -1;
			area2ArtifactsPercentageNewEB = -1;
			area3ArtifactsPercentageNewEB = -1;
			area4ArtifactsPercentageNewEB = -1;


	}

	char ucChannelName[20];
	int	nZeroValuesInserted;
	int nInvalid;
	float allArtifactPercentage;
	float allArtifactPercentageNewEB;
	float otherArtifactsPercentage;
	float EBPercentage;
	float NewEBPercentage;
	float SatPercentage;
	float SpkPercentage;
	float ExcPercentage;
	float MBPercentage;
	float area1ArtifactsPercentage;
	float area2ArtifactsPercentage;
	float area3ArtifactsPercentage;
	float area4ArtifactsPercentage;
	float area1ArtifactsPercentageNewEB;
	float area2ArtifactsPercentageNewEB;
	float area3ArtifactsPercentageNewEB;
	float area4ArtifactsPercentageNewEB;

	

	

};

struct BALERT_EPOCH_ARTIF_INFO
{
public:
	BALERT_EPOCH_ARTIF_INFO()
	{
		memset(ucChannelName,0, sizeof(char)*20);
		nInvalid = DEFAULT_VALUE;
		//nEMGLevel = DEFAULT_VALUE;
		nZeroValuesInserted = DEFAULT_VALUE;
		nEbEffected = DEFAULT_VALUE;
		nSatEffected  = DEFAULT_VALUE;
		nExcEffected = DEFAULT_VALUE;
		nSpkEffected  = DEFAULT_VALUE;
		nMissedBlock  = DEFAULT_VALUE;
		nPeriodicMissedBlock = DEFAULT_VALUE;
	}

	char ucChannelName[20];
	int nInvalid;
	//int	nEMGLevel;
	int	nZeroValuesInserted;
	int nEbEffected;
	int nSatEffected;
	int nExcEffected;
	int nSpkEffected;
	int	nMissedBlock;
	int nPeriodicMissedBlock;

};

struct BALERT_OVERLAYS_ARTIF_INFO
{
public:
	BALERT_OVERLAYS_ARTIF_INFO()
	{
		memset(ucChannelName,0, sizeof(char)*20);
		nInvalid = DEFAULT_VALUE;
		//nEMGLevel = DEFAULT_VALUE;
		nZeroValuesInserted = DEFAULT_VALUE;
		nEbEffected = DEFAULT_VALUE;
		nSatEffected  = DEFAULT_VALUE;
		nExcEffected = DEFAULT_VALUE;
		nSpkEffected  = DEFAULT_VALUE;
		nMissedBlock  = DEFAULT_VALUE;
		nPeriodicMissedBlock = DEFAULT_VALUE;
	}

	char ucChannelName[20];
	int	nInvalid;
	//int	nEMGLevel;
	int	nZeroValuesInserted;
	int nEbEffected;
	int nSatEffected;
	int nExcEffected;
	int nSpkEffected;
	int	nMissedBlock;
	int nPeriodicMissedBlock;

};

#define		SPIKE_CODE	0
#define		EXCURSION_CODE	1
#define		SATURATION_CODE	2

#define		MISSED_BLOCKS_CODE	4
#define		EYE_BLINK_CODE		5
#define		PERIODIC_IMP_MISSED_BLOCKS_CODE	6
