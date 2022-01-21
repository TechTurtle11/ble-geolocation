#!/bin/bash
#Run as root reference bluetooth core specification for more details vol 4 ,7.8.5


if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi


hciconfig hci0 down
hciconfig hci0 up
#advertising packet data
hcitool -i hci0 cmd 0x08 0x0008 1E 02 01 1A 1A FF 4C 00 02 15 63 6F 3F 8F 64 91 4B EE 95 F7 D8 CC 64 A8 63 B5 00 00 00 00 C8 00
# sets advertsing rate to 100ms 
hcitool -i hci0 cmd 0x08 0x0006 A0 00 A0 00 00 00 00 00 00 00 00 00 00 07 00
#LE Set Advertise Enable
hcitool -i hci0 cmd 0x08 0x000A 01
hciconfig hci0 noscan
