import struct
import sys
import math

import numpy as np


#
# TODO 1: (NOW FIXED) Find the first occurrence of magic and start from there
# TODO 2: Warn if we cannot parse a specific section and try to recover
# TODO 3: Remove error at end of file if we have only fragment of TLV
#


def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('2I', data)
    return tlvType, tlvLength


def parseDetectedObjects(data, numObj, tlvLength):
    detected_points = struct.unpack(str(numObj * 4) + 'f', data[:tlvLength])
    detected_points = np.asarray(detected_points).reshape(numObj, 4)
    return detected_points


def parseRangeProfile(data, tlvLength):
    # an integer is 2 byte long
    range_bins = tlvLength / 2
    range_profile = struct.unpack(str(int(range_bins)) + 'H', data[:tlvLength])
    return range_profile


def parseRDheatmap(data, tlvLength, range_bins, rm_clutter=True):
    """
    range bins times doppler bins times 2, doppler bins = chirps/ frame divided by num of antennas TX (3)
    #default chirps per frame is (128/3) = 42 * 2 * 256

    the call to replace_left_right mirror-flips left and right after reshaping.
    replace_left_right is equivalent to this line from mmWave.js in the visualizer code
    # rangeDoppler = rangeDoppler.slice((rangeDoppler.length + 1) / 2).concat(
    #     rangeDoppler.slice(0, (rangeDoppler.length + 1) / 2));

    :param range_bins:
    :param data: the incoming byte stream to be interpreted as range-doppler heatmap/profile
    :param tlvLength:
    :return:
    """
    doppler_bins = (tlvLength / 2) / range_bins
    rd_heatmap = struct.unpack(str(int(range_bins * doppler_bins)) + 'H', data[:tlvLength])
    rd_heatmap = np.reshape(rd_heatmap, (int(range_bins), int(doppler_bins)))

    overall_mean = np.mean(rd_heatmap)
    if rm_clutter:
        rd_heatmap = np.array([row - np.mean(row) for row in rd_heatmap])

    return replace_left_right(rd_heatmap)


def chg_val(val):
    return val - 65536 if val > 32767 else val


def parseAziheatmap(data, tlvLength, range_bins):
    """
    :param range_bins:
    :param data: the incoming byte stream to be interpreted as range-doppler heatmap/profile
    :param tlvLength:
    :return:
    """
    # range_bins = 256
    azi_bins = (tlvLength / 2) / range_bins
    azi_heatmap = struct.unpack(str(int(range_bins * azi_bins)) + 'H', data[:tlvLength])
    # azi_heatmap = [chg_val(x) for x in azi_heatmap]
    azi_heatmap = np.reshape(azi_heatmap, (int(range_bins), int(azi_bins)))

    # use the default order of 3 Tx's and ordering is TX0, TX1, TX2
    row_indices = [7, 5, 11, 9]
    qrows = 4
    qcols = range_bins
    rowSizeBytes = 48
    q = data[:tlvLength]
    qq = []
    for col in range(qcols):
        real = []
        img = []
        for row in range(qrows):
            index = col * rowSizeBytes + 4 * row_indices[row]
            real.append(q[index + 1] * 256 + q[index])
            img.append(q[index + 3] * 256 + q[index + 2])
        real = [chg_val(x) for x in real]
        img = [chg_val(x) for x in img]
        # convert to complex numbers
        data = np.array([real, img]).transpose()
        data = np.pad(data, ((0, 60), (0, 0)), 'constant', constant_values=0)
        data = data[..., 0] + 1j * data[..., 1]
        transformed = np.fft.fft(data)

        # take the magnitude
        transformed = np.absolute(transformed)
        qq.append(np.concatenate((transformed[int(len(transformed) / 2):], transformed[:int(len(transformed) / 2)])))
    qq = np.array(qq)
    return qq


def replace_left_right(a):
    rtn = np.empty(shape=a.shape)
    rtn[:, :int(rtn.shape[1] / 2)] = a[:, int(rtn.shape[1] / 2):]
    rtn[:, int(rtn.shape[1] / 2):] = a[:, :int(rtn.shape[1] / 2)]
    return rtn


def parseStats(data):
    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack('6I', data[:24])
    return interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad
    # print("\tOutputMsgStats:\t%d " % (6))
    # print("\t\tChirpMargin:\t%d " % (chirpMargin))
    # print("\t\tFrameMargin:\t%d " % (frameMargin))
    # print("\t\tInterCPULoad:\t%d " % (interCPULoad))
    # print("\t\tActiveCPULoad:\t%d " % (activeCPULoad))
    # print("\t\tTransmitOut:\t%d " % (transmitOut))
    # print("\t\tInterprocess:\t%d " % (interProcess))


negative_rtn = False, None, None, None, None, None


class tlv_header_decoder():
    def __init__(self):
        pass


def decode_iwr_tlv(in_data):
    """
    Must disable range profile for the quick RD heatmap to work, this way the number of range bins will be be calculated
    from the absent range profile. You can still get the range profile by inferring it from the RD heatmap
    :param in_data:
    :return: if no detected point at this frame, the detected point will be an empty a
    """
    magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    header_length = 36

    offset = in_data.find(magic)
    data = in_data[offset:]
    if len(data) < header_length:
        return negative_rtn
    try:
        data_magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('Q7I',
                                                                                                    data[
                                                                                                    :header_length])
    except struct.error:
        print("Improper TLV structure found: ", (data,))
        return negative_rtn
    # print("Packet ID:\t%d "%(frameNum))
    # print("Version:\t%x "%(version))
    # print("Data Len:\t\t%d", length)
    # print("TLV:\t\t%d "%(numTLVs))
    # print("Detect Obj:\t%d "%(numObj))
    # print("Platform:\t%X "%(platform))
    if version >= 50462726 and len(data) >= length:
        # if version > 0x01000005 and len(data) >= length:
        try:
            sub_frame_num = struct.unpack('I', data[36:40])[0]
            header_length = 40
            # print("Subframe:\t%d "%(subFrameNum))
            pending_bytes = length - header_length
            data = data[header_length:]

            detected_points = None
            range_profile = None
            rd_heatmap = None
            azi_heatmap = None
            range_bins = 8
            statistics = None

            for i in range(numTLVs):
                tlvType, tlvLength = tlvHeaderDecode(data[:8])
                data = data[8:]
                if tlvType == 1:
                    # print('Outputting Points')
                    detected_points = parseDetectedObjects(data, numObj,
                                                           tlvLength)  # if no detected points, tlvType won't have 1
                elif tlvType == 2:
                    # the range bins is modified in the range profile is enabled
                    range_profile = parseRangeProfile(data, tlvLength)

                elif tlvType == 4:
                    # resolving static azimuth heatmap
                    pass
                elif tlvType == 5:
                    # try:
                    #     assert range_bins
                    # except AssertionError:
                    #     raise Exception('Must enable range-profile while enabling range-doppler-profile, in order to'
                    #                     'interpret the number of range bins')
                    rd_heatmap = parseRDheatmap(data, tlvLength, range_bins)
                elif tlvType == 6:
                    # TODO why is the states' TLV not present?
                    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = parseStats(data)
                    pass
                elif tlvType == 7:
                    pass
                elif tlvType == 8:
                    # resolving static azimuth-elevation heatmap
                    try:
                        azi_heatmap = parseAziheatmap(data, tlvLength, range_bins)
                    except:
                        print('bad azimuth')
                        azi_heatmap = None
                    pass
                elif tlvType == 9:  # only for AoP EV2
                    pass
                else:
                    # print("Unidentified tlv type %d" % tlvType, '. Its len is ' + str(tlvLength))
                    n_offset = data.find(magic)
                    if n_offset != offset and n_offset != -1:
                        print('New magic found, discarding previous frame with unknown tlv')
                        data = data[n_offset:]
                        return True, data, detected_points, range_profile, rd_heatmap, azi_heatmap
                data = data[tlvLength:]
                pending_bytes -= (8 + tlvLength)
            data = data[pending_bytes:]  # data that are left

            # infer range profile from heatmap is the former is not enabled
            if range_profile is None and rd_heatmap is not None and len(rd_heatmap) > 0:
                range_profile = rd_heatmap[:, 0]
            return True, data, detected_points, range_profile, rd_heatmap, azi_heatmap
        except struct.error as se:
            print('Failed to parse tlv message, type = ' + str(tlvType) + ', error: ')
            print(se)
            pass

    return negative_rtn


if __name__ == "__main__":
    magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    fileName = 'D:/PycharmProjects/mmWave_gesture_iwr6843/test_data2.dat'
    rawDataFile = open(fileName, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()
    offset = rawData.find(magic)
    rawData = rawData[offset:]

    # for i in range(len(rawData/36))
    #
    # for length, frameNum in tlvHeader(rawData):
    #     print
