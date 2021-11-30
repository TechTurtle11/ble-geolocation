import subprocess
import parse

BEACON_PREFIX = "geolocation_beacon"


class BluetoothService:

    def __init__(self) -> None:

        process = subprocess.run(
            ["btmgmt", "find"], capture_output=True, encoding="utf-8", text=True)

        if process.returncode == 0:
            device_strings = process.stdout.replace(
                "\n", "").split("hci0 dev_found: ")[1:]
            # print(process.stdout)

            c = parse.compile("\nhci0 dev_found: {address} type {type:S} rssi {rssi:d} flags {flags:x} \nname {name:S}")
            for each in c.findall(process.stdout):
                print(each)
            devices = [self.parse_device_string(
                device_string) for device_string in device_strings]
            # 'print(devices)
        else:
            print(f"bluetooth was unable to scan for devices")

    def parse_device_string(self, device_string):
        device = {}
        words = device_string.split()
        # device["address"] = words[0]
        device["address"] = ":".join(parse.search("{}:{}:{}:{}:{}:{} ", device_string).fixed)
        device["type"] = parse.search("type {:S} ", device_string).fixed[0]
        device["rssi"] = parse.search("rssi {:d} ", device_string).fixed[0]
        name_parse = parse.search("name {:S}", device_string)
        device["name"] = name_parse.fixed[0] if isinstance(name_parse, parse.Result) else None

        return device


BluetoothService()
