import brainflow
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamOutlet
import serial
from serial.tools import list_ports


class OpenBCIToLSLInterface:
    _BOARD_ALIASES = {
        "cyton": BoardIds.CYTON_BOARD.value,
        "cyton8": BoardIds.CYTON_BOARD.value,
        "cytondaisy": BoardIds.CYTON_DAISY_BOARD.value,
        "daisy": BoardIds.CYTON_DAISY_BOARD.value,
    }
    _PORT_HINTS = ("openbci", "usb serial", "ftdi", "cp210", "ch340")

    def __init__(
        self,
        stream_name,
        stream_type="EEG",
        serial_port="COM5",
        board_id=BoardIds.CYTON_BOARD.value,
        log=False,
        streamer_params="",
        ring_buffer_size=45000,
    ):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.params.ip_port = 0
        self.params.mac_address = ""
        self.params.other_info = ""
        self.params.serial_number = ""
        self.params.ip_address = ""
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = ""

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.board_id = self._resolve_board_id(board_id)
        self.streamer_params = streamer_params
        self.ring_buffer_size = ring_buffer_size
        self.board_descr = BoardShim.get_board_descr(self.board_id)
        self.channel_count = 0
        self.info_eeg = None
        self.outlet_eeg = None

        if log:
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        try:
            self.board = BoardShim(self.board_id, self.params)
        except brainflow.board_shim.BrainFlowError as error:
            raise RuntimeError(
                f"Cannot create a BrainFlow board for board_id={self.board_id} on {serial_port}."
            ) from error

    @classmethod
    def _resolve_board_id(cls, board_id):
        if isinstance(board_id, int):
            return board_id

        if isinstance(board_id, str):
            stripped = board_id.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)

            normalized = "".join(character for character in stripped.lower() if character.isalnum())
            if normalized in cls._BOARD_ALIASES:
                return cls._BOARD_ALIASES[normalized]

        raise ValueError(
            f"Unsupported board_id {board_id!r}. Use 0/'cyton' or 2/'cyton+daisy'."
        )

    @staticmethod
    def _get_serial_ports():
        return list(list_ports.comports())

    @classmethod
    def _format_serial_ports(cls):
        ports = cls._get_serial_ports()
        if not ports:
            return "No serial ports detected."

        lines = []
        for port in ports:
            parts = [port.device, port.description]
            if port.manufacturer:
                parts.append(f"manufacturer={port.manufacturer}")
            if port.serial_number:
                parts.append(f"serial={port.serial_number}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    @classmethod
    def _find_openbci_port(cls):
        ports = cls._get_serial_ports()
        for port in ports:
            searchable = " ".join(
                value for value in (port.device, port.description, port.manufacturer or "") if value
            ).lower()
            if any(hint in searchable for hint in cls._PORT_HINTS):
                return port.device
        return ports[0].device if ports else None

    def _resolve_serial_port(self):
        port = (self.params.serial_port or "").strip()
        if port and port.upper() != "AUTO":
            return port

        detected_port = self._find_openbci_port()
        if detected_port is None:
            raise RuntimeError(
                "Unable to detect an OpenBCI serial port.\n"
                f"Detected ports:\n{self._format_serial_ports()}"
            )
        self.params.serial_port = detected_port
        return detected_port

    def _assert_serial_port_accessible(self):
        port = self._resolve_serial_port()
        try:
            probe = serial.Serial(port=port, timeout=1)
            probe.close()
        except serial.SerialException as error:
            error_message = str(error)
            if "Access is denied" in error_message:
                raise RuntimeError(
                    f"Serial port {port} exists but is already in use by another application.\n"
                    "Close any program that may still own the dongle, such as another Python process, "
                    "OpenBCI GUI, Arduino Serial Monitor, PuTTY, or a previous PhysioLabXR run.\n"
                    f"Detected ports:\n{self._format_serial_ports()}"
                ) from error
            raise RuntimeError(
                f"Unable to open serial port {port}: {error_message}\n"
                f"Detected ports:\n{self._format_serial_ports()}"
            ) from error

    def _build_channel_descriptors(self):
        eeg_labels = [
            label.strip()
            for label in self.board_descr.get("eeg_names", "").split(",")
            if label.strip()
        ]
        accel_count = len(self.board_descr.get("accel_channels", []))
        other_count = len(self.board_descr.get("other_channels", []))
        analog_count = len(self.board_descr.get("analog_channels", []))

        descriptors = [("PackageNum", "counter", "COUNTER")]
        descriptors.extend((label, "microvolts", "EEG") for label in eeg_labels)

        if accel_count == 3:
            descriptors.extend(
                (label, "g", "ACC") for label in ("X", "Y", "Z")
            )
        else:
            descriptors.extend(
                (f"Accel{index + 1}", "g", "ACC") for index in range(accel_count)
            )

        descriptors.extend(
            (f"Other{index + 1}", "aux", "AUX") for index in range(other_count)
        )
        descriptors.extend(
            (f"Analog{index + 1}", "aux", "AUX") for index in range(analog_count)
        )
        descriptors.append(("TimeStamp", "seconds", "TIME"))
        descriptors.append(("Marker", "marker", "MARKER"))
        return descriptors

    def start_sensor(self):
        self._assert_serial_port_accessible()
        try:
            self.board.prepare_session()
        except brainflow.board_shim.BrainFlowError as error:
            raise AssertionError(
                "Unable to connect to device: please check the sensor connection, the COM port, and the board_id."
            ) from error
        print(
            f"OpenBCIInterface: connected to sensor on {self.params.serial_port} with board id {self.board_id}."
        )

        try:
            self.board.start_stream(self.ring_buffer_size, self.streamer_params)
        except brainflow.board_shim.BrainFlowError as error:
            raise AssertionError(
                "Unable to start streaming: please verify the OpenBCI board is powered on and the board_id matches the hardware."
            ) from error
        print("OpenBCIInterface: started streaming.")

        self.create_lsl()

    def process_frames(self):
        frames = self.board.get_board_data()
        if frames.size == 0:
            return frames

        for frame in frames.T:
            self.push_frame(frame)

        return frames

    def stop_sensor(self):
        try:
            if self.board.is_prepared():
                try:
                    self.board.stop_stream()
                    print("OpenBCIInterface: stopped streaming.")
                except brainflow.board_shim.BrainFlowError:
                    pass

                self.board.release_session()
                print("OpenBCIInterface: released session.")
        except brainflow.board_shim.BrainFlowError as error:
            print(error)

    def create_lsl(self, name=None, type=None, nominal_srate=None, channel_format="float32", source_id=None):
        name = name or self.stream_name
        stream_type = type or self.stream_type
        nominal_srate = nominal_srate or self.board_descr["sampling_rate"]
        self.channel_count = self.board_descr["num_rows"]
        source_id = source_id or f"OpenBCI_{self.board_descr['name']}_{self.board_id}"

        self.info_eeg = StreamInfo(
            name=name,
            type=stream_type,
            channel_count=self.channel_count,
            nominal_srate=nominal_srate,
            channel_format=channel_format,
            source_id=source_id,
        )

        chns = self.info_eeg.desc().append_child("channels")
        channel_descriptors = self._build_channel_descriptors()
        if len(channel_descriptors) != self.channel_count:
            raise RuntimeError(
                f"Board description generated {len(channel_descriptors)} labels for a {self.channel_count}-channel stream."
            )

        for label, unit, channel_type in channel_descriptors:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", unit)
            ch.append_child_value("type", channel_type)

        self.info_eeg.desc().append_child_value("manufacturer", "OpenBCI Inc.")
        self.outlet_eeg = StreamOutlet(self.info_eeg)

        print(
            "--------------------------------------\n"
            "LSL Configuration:\n"
            "  Stream 1:\n"
            f"      Name: {name}\n"
            f"      Type: {stream_type}\n"
            f"      Board Profile: {self.board_descr['name']} ({self.board_id})\n"
            f"      Channel Count: {self.channel_count}\n"
            f"      Sampling Rate: {nominal_srate}\n"
            f"      Channel Format: {channel_format}\n"
            f"      Source Id: {source_id}\n"
        )

    def push_frame(self, samples):
        if self.outlet_eeg is None:
            raise RuntimeError("LSL outlet is not ready. Call start_sensor() or create_lsl() first.")

        if hasattr(samples, "tolist"):
            samples = samples.tolist()
        else:
            samples = list(samples)

        if len(samples) != self.channel_count:
            raise ValueError(
                f"LSL sample length mismatch: got {len(samples)} values, but the outlet expects {self.channel_count}. "
                f"Check board_id (Cyton=0, Cyton+Daisy=2)."
            )

        self.outlet_eeg.push_sample(samples)

    def info_print(self):
        print("Board Information:")
        print("Sampling Rate:", self.board_descr["sampling_rate"])
        print("Board Id:", self.board_id)
        print("EEG names:", self.board_descr["eeg_names"])
        print("Package Num Channel: ", self.board_descr["package_num_channel"])
        print("EEG Channels:", self.board_descr["eeg_channels"])
        print("Accel Channels: ", self.board_descr["accel_channels"])
        print("Other Channels:", self.board_descr["other_channels"])
        print("Analog Channels: ", self.board_descr["analog_channels"])
        print("TimeStamp: ", self.board_descr["timestamp_channel"])
        print("Marker Channel: ", self.board_descr["marker_channel"])


def run_test_lsl(openbci_interface):
    while True:
        try:
            openbci_interface.process_frames()
        except KeyboardInterrupt:
            print("Stopped streaming.")
            break
    return True


def main():
    openbci_interface = OpenBCIToLSLInterface(
        stream_name="OpenBCI_Cyton_Daisy_15_Channels",
        stream_type="EEG",
        serial_port="COM4",
        board_id="cyton+daisy",
        log=False,
        ring_buffer_size=45000,
    )

    try:
        openbci_interface.start_sensor()
        run_test_lsl(openbci_interface)
    finally:
        openbci_interface.stop_sensor()


if __name__ == "__main__":
    main()
