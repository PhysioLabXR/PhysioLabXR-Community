
#ifndef   __EEGACQ_DEFINE__
#define   __EEGACQ_DEFINE__


#define     MAX_NUM_ELECTRODE				24//fz,cz,po,veog,heog,ref
#define		MAX_NUM_EEGCHANNELS				24
#define     MAX_NUM_ALLCHANNELS				26 //the number of channels including time channels (X24 has 24 eeg + 2 time ch)


#define		ABM_SESSION_RAW			0	// - gives RAW and RAW PSD data
#define		ABM_SESSION_DECON		1	//- gives  all of ABM_SESSION_RAW plus  DECON and DECON PSD
#define		ABM_SESSION_BSTATE		2	//- gives all of ABM_SESSION_DECON plus BSTATE
#define		ABM_SESSION_WORKLOAD	3	//- gives all of  ABM_SESSION_BSTATE  plus WORKLOAD

// device codes are changed according to codes in Rome

#define		ABM_DEVICE_X24Flex_10_20    6
#define		ABM_DEVICE_X10Flex_Standard			7
#define		ABM_DEVICE_X24Flex_Reduced			8
#define		ABM_DEVICE_X24Flex_10_20_LM		9
#define		ABM_DEVICE_X24LE_10_20_LM		10
#define		ABM_DEVICE_X24Flex_10_20_LM_Red			11
#define		ABM_DEVICE_X24LE_10_20_LM_Red		12
#define		ABM_DEVICE_X10Flex_10_20_LM_Red			13
#define		ABM_DEVICE_X24LE_Ambulatory			14

#define		IS_ABM_DEVICE_X24(x) (x==ABM_DEVICE_X24Flex_10_20 || x==ABM_DEVICE_X24Flex_Reduced || x==ABM_DEVICE_X24Flex_10_20_LM || x==ABM_DEVICE_X24LE_10_20_LM || x==ABM_DEVICE_X24Flex_10_20_LM_Red || x==ABM_DEVICE_X24LE_10_20_LM_Red || x==ABM_DEVICE_X24LE_Ambulatory)
#define		IS_ABM_DEVICE_X10(x) (x==ABM_DEVICE_X10Flex_Standard || x==ABM_DEVICE_X10Flex_10_20_LM_Red)
#define		IS_ABM_DEVICE_24_CH(x) (x==ABM_DEVICE_X24Flex_10_20 || x==ABM_DEVICE_X24Flex_10_20_LM || x==ABM_DEVICE_X24LE_10_20_LM || x==ABM_DEVICE_X24LE_Ambulatory)
#define		IS_ABM_DEVICE_10_CH(x) (x==ABM_DEVICE_X10Flex_Standard || x==ABM_DEVICE_X24Flex_Reduced || x==ABM_DEVICE_X24Flex_10_20_LM_Red || x==ABM_DEVICE_X24LE_10_20_LM_Red || x==ABM_DEVICE_X10Flex_10_20_LM_Red)
#define		IS_ABM_DEVICE_24LE(x) (x==ABM_DEVICE_X24LE_10_20_LM || x==ABM_DEVICE_X24LE_10_20_LM_Red || x==ABM_DEVICE_X24LE_Ambulatory)

#define		ABM_THIRD_PARTY_PORTS_NUM  3

#define		ESU_TYPE_UNKNOWN			0
#define		ESU_TYPE_DONGLE				1
#define		ESU_TYPE_SINGLE_CHANNEL		2
#define		ESU_TYPE_MULTI_CHANNEL		3


#define     IMPEDANCE_REFERENTIAL	0
#define     IMPEDANCE_DIFERENTIAL	1
#define     IMPEDANCE_NOT_AVAILABLE	2
#define     IMPEDANCE_REFERENTIAL_SECONDARY 10


// windows messages for communication between threads
#define		WM_DATARECEIVED		     WM_USER
#define		WM_COMMANDRECEIVED	    (WM_USER + 1)
#define		WM_NOTRECEIVING		    (WM_USER + 2)
#define		WM_ANSWERECEIVED		    (WM_USER + 3)
#define		WM_DEVCHANGEFREQ		    (WM_USER + 4)
#define		WM_TECHNICAL_MON_RESET	 (WM_USER + 5)
#define		WM_AMP_ABD_LINK_CHECKED	 (WM_USER + 6)
#define     WM_UPDATE_COMMUNICATION_PORTS (WM_USER + 7)
#define     WM_IMPEDANCE_MSG         (WM_USER + 8)
#define     WM_TM_FINISHED_MSG       (WM_USER + 9)
#define     WM_TM_START_MSG			 (WM_USER + 10)
#define     WM_IMPEDANCE_START_MSG	 (WM_USER + 11)


#define		SDK_WAITING_MODE           -1
#define		SDK_NORMAL_MODE            0
#define		SDK_IMPEDANCE_MODE         1
#define		SDK_TECHNICALMON_MODE      2

#define		MAX_LENGTH_CHANNEL_NAME		20

#define		TIMESTAMP_RAW				0
#define		TIMESTAMP_PSD				1
#define		TIMESTAMP_DECON				2
#define		TIMESTAMP_CLASS				3
#define		TIMESTAMP_EKG				4
#define		TIMESTAMP_PSDRAW			5
#define		TIMESTAMP_MOVEMENT			6
#define		TIMESTAMP_BANDOVERPSD		7
#define		TIMESTAMP_PSDBANDWIDTH		8
#define		TIMESTAMP_PSDBANDWIDTHRAW	9
#define		TIMESTAMP_ZSCORE			10
#define		TIMESTAMP_PULSERATE			11
#define		TIMESTAMP_FILTERED			12
#define		TIMESTAMP_RAWRAW			13
#define		TIMESTAMP_BANDOVERPSDRAW	14
#define		TIMESTAMP_RAWTILTS			15
#define		TIMESTAMP_ANGLES			16
#define		TIMESTAMP_OPTICAL			17
#define		TIMESTAMP_HAPTIC			18
#define		TIMESTAMP_QUALITY			19

#define		TIMESTAMP_RAW_NEW			100
#define		TIMESTAMP_NEW				120

#define		ESU_TIMESTAMP_LENGTH		4
#define		SYSTEM_TIMESTAMP_LENGTH		8

///block 1 host
//#define     POS_SERIAL_NUMBER_H      0
//#define     POS_HARDWARE_VERSION_H   16
//#define     POS_FIRMWARE_VERSION_H   20
////
//////block 2 host
//#define     POS_SYSSERIAL_NUMBER_H   0
//#define     POS_RELEASE_H            16
////
//////block 3 host
//#define     POS_TX_OFFSET_H          0
//#define     POS_RX_OFFSET_H          2
//#define     POS_DEFF_FRQ_CH_H        4
//#define     POS_CONFIG_WORD_H        6
//#define     POS_EEPROM_HOST          8
//
//
////block 1 acq
//#define     POS_SERIAL_NUMBER_A      0
//#define     POS_HARDWARE_VERSION_A   16
//#define     POS_FIRMWARE_VERSION_A   20
////
//////block 2 acq
//#define     POS_SYSSERIAL_NUMBER_A   0
//#define     POS_RELEASE_A            16
////
//////block 3 acq
//#define     POS_TX_OFFSET_A          0
//#define     POS_RX_OFFSET_A          2
//#define     POS_DEFF_FRQ_CH_A        4
//#define     POS_TILT_ACC	         5
//#define     POS_CONFIG_WORD_A        6
//#define     POS_EEPROM_ACQ           8
////
//
////block 4 adc
//#define     POS_ANLG_BRD_SER_NUM_ADC 0
//#define     POS_HARDWARE_VERSION_ADC 16
//#define     POS_HEADSET              20
//#define     POS_ADC_BITSPERCH        22
//#define     POS_ECG                  23
////
//////block 5 adc
//#define     POS_CHANN_CONFIG_ADC     0
//#define     POS_DC_OFFSET_ADC        12
////
//#define     INDEX_BLOCK_GAIN         6
//#define     INDEX_BLOCK_VERSION      1
//////block 6 adc
//#define     POS_GAIN_CONSTS_ADC      0
//#define     POS_IMP_CONSTS_ADC       12
//
//#define     SEND_OFFLINE_SLEEP       60
//
////degug file pointers
//#define     ACQ_MESSAGE_COUNTER      0
//#define     ACQ_BATTERY              1
//#define     ACQ_VCC                  2
//#define     ACQ_RSSI                 3
//#define     ACQ_TEMP                 4
//#define     ACQ_RECC_ERR_LAST        5
//#define     ACQ_TRSM_ERR_LAST        6
//#define     ACQ_TRSM_CHNO            7
//#define     ACQ_RECC_CHNO            8
//#define     ACQ_IMP_LEV              9
//#define     ACQ_IMP_CHANNEL          10
//#define     ACQ_SIMULATED_FLG        11
//#define     ACQ_FRQRXOFFSET          12
//#define     HOST_BATTERY             13
//#define     HOST_VCC                 14
//#define     HOST_RSSI                15
//#define     HOST_TEMP                16
//#define     HOST_RECC_ERR_LAST       17
//#define     HOST_TRSM_ERR_LAST       18
//#define     HOST_TRSM_CHNO           19
//#define     HOST_RECC_CHNO           20
//#define     HOST_SIMULATED_FLG       21
//#define     HOST_FRQRXOFFSET         22
//
#define     DC_OFFSET_RT             49

////eprom
//#define     EEPROM_NOT_EXIST         0
//#define     EEPROM_EXIST_NOT_INIT    1
//#define     EEPROM_EXIST_INIT        2
//
////one minute
//#define     ONE_MINUTE_SEC           60
//
//////type of analog boards
//#define     TYPE_OLDEST_ANALOG_BOARD    0
//#define     TYPE_MEDIUM_ANALOG_BOARD    1
//#define     TYPE_NEW_ANALOG_BOARD       2
//#define     TYPE_7_EL_ANALOG_BOARD      3
//
//#define     TYPE_ANALOG_BOARD_July05    5
//#define     TYPE_ANALOG_BOARD_209_210   6
//#define     TYPE_ANALOG_BOARD_206       7
//#define     TYPE_ANALOG_BOARD_224       8
//#define     TYPE_ANALOG_BOARD_9_CHANNEL 9
//

//#define     VCP_SERIAL_PORT          0
//#define     USB_DESCRIPTION_PORT     1

#define     MAX_NUMBER_PACKETS       65536

#endif //__EEGACQ_DEFINE__

