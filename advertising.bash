#!/bin/bash

#Run as root reference bluetooth core specification for more details vol 4 ,7.8.5
#https://stackoverflow.com/questions/21124993/is-there-a-way-to-increase-ble-advertisement-frequency-in-bluez


hciconfig hci0 up
#advertising packet data
hcitool -i hci0 cmd 0x08 0x0008 1E 02 01 06 1A FF 4C 00 02 15 C7 C1 A1 BF BB 00 4C AD 87 04 9F 2D 29 17 DE D2 00 00 00 00 C8 00
# sets advertsing rate to 100ms 
hcitool -i hci0 cmd 0x08 0x0006 A0 00 A0 00 03 00 00 00 00 00 00 00 00 07 00
#LE Set Advertise Enable
hcitool -i hci0 cmd 0x08 0x000a 01