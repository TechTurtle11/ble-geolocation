import subprocess


BEACON_PREFIX = "geolocation_beacon"


class BluetoothService():

    def __init__(self) -> None:

        process = subprocess.run(
            ["btmgmt", "find"], capture_output=True, encoding="utf-8", text=True)

        if process.returncode == 0:
            device_strings = process.stdout.replace(
                "\n", "").split("hci0 dev_found: ")[1:]
            devices = [self.parse_device_string(
                device_string) for device_string in device_strings]
            print(devices)
        else:
            print(f"bluetooth was unable to scan for devices")

    def parse_device_string(self, device_string):
        device = {}
        words = device_string.split()
        device["address"] = words[0]
        device["type"] = words[2]
        device["rssi"] = int(words[5])
        try:
            index = words.index("name")
            device["name"] = " ".join(words[index+1:])
        except ValueError:
            device["name"] = "none"
        return device


BluetoothService()
