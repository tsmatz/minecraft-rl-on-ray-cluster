#!/bin/bash

VAR_PATH="${MALMO_XSD_PATH}/../Minecraft"
cd $VAR_PATH
#./launchClient.sh -port $1
xvfb-run -a -e /dev/stdout -s '-screen 0 640x480x16' ./launchClient.sh -port $1
