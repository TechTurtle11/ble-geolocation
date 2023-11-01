#!/bin/bash

set -e


docker compose build
docker compose run --user "$(id -u):$(id -g)"  --workdir=$pwd dev-image bash