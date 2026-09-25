from pedalboard.io import AudioStream
from pedalboard import Distortion, Chorus, LadderFilter

# Set the input and output devices
input_device = "USB Audio CODEC "
output_device = "MacBook Pro Speakers"

wah = LadderFilter(
    mode = LadderFilter.Mode.BPF12,
    cutoff_hz = 400,
    resonance = 0.7
)

with AudioStream(
    input_device_name=input_device,
    output_device_name=output_device
) as stream:

    stream.plugins.append(wah)

    input("Wah LOW - press Enter...")
    wah.cutoff_hz = 400

    input("Wah MID - press Enter...")
    wah.cutoff_hz = 1200

    input("Wah HIGH - press Enter...")
    wah.cutoff_hz = 2500

    input("Press Enter to stop...")
