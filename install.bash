

#!/bin/bash
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi


apt-get install python3 python3-dev


python3 -m venv geolocation
source geolocation/bin/activate
pip3 install -r requirements.txt