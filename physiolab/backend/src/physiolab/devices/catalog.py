from physiolab.contracts import Catalog, DeviceInfo

CATALOG = Catalog(
    devices=[
        DeviceInfo(
            id="simulated-emg",
            name="Simulated EMG",
            delivery="simulated",
            status="ready",
            notes="100 Hz, one channel, arbitrary units. Deterministic development fixture.",
        ),
        DeviceInfo(
            id="apple-watch",
            name="Apple Watch",
            delivery="unverified",
            status="planned",
            notes=(
                "Serena: validate HealthKit/companion app permissions, "
                "data types and live availability."
            ),
        ),
        DeviceInfo(
            id="oura",
            name="Oura Ring",
            delivery="batch",
            status="planned",
            notes=(
                "Serena: OAuth V2 connector; expose actual sync timestamps, "
                "not a live stream badge."
            ),
        ),
        DeviceInfo(
            id="phone-motion",
            name="Phone accelerometer",
            delivery="unverified",
            status="planned",
            notes="Serena: local bridge, permissions, measured sampling rate and timestamps.",
        ),
        DeviceInfo(
            id="legacy-lsl",
            name="Existing LSL / EEG connectors",
            delivery="unverified",
            status="legacy-adapter",
            notes="Adapter seam exists; hardware and clock conversion need validation.",
        ),
    ],
    processors=["bandpass", "sliding-window"],
    models=["demo-emg-rms"],
    targets=["web-preview"],
)
