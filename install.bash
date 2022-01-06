

#!/bin/bash
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

#!/bin/bash
apt-get install python3 python3-dev bluez

pip3 install -r requirements.txt