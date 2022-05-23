#!/bin/bash

### carla_docker.sh -- run Carla environment in Docker container
###
### Usage:
###   ./carla_docker.sh [options]
###
### Options:
###   -h    Show this message
###   -o    Offscreen mode: disable CARLA server window
###

usage() {
    # Use the file header as a usage guide
    # Reference: https://samizdat.dev/help-message-for-shell-scripts/
    sed -rn 's/^### ?/ /;T;p' "$0"
}

version() {
    # Convert decimal version numbers into integers suitable for comparison
    echo "$@" | awk -F. '{ printf("%03d%03d%03d\n", $1,$2,$3); }'
}


# Get directory of scripts
script_dir=$(dirname "$(readlink -f "$0")")

# Set name of CARLA parameters
carla_image="carlasim/carla:0.9.6"
carla_egg_dir="/home/carla/PythonAPI/carla/dist"

# Set nvidia container runtime flag (dependent on docker version)
if [ "$(version "$(docker version -f '{{.Server.Version}}')")" -gt "$(version 19.03.0)" ]; then
    nvidia_flag="--gpus all"
else
    nvidia_flag="--runtime=nvidia"
fi

# Set SDL_VIDEODRIVER env var to make CARLA server visible
sdl_driver=x11

# Parse arguments with posix-compatible getopt
cmdargs=$(GETOPT_COMPATIBLE=1 getopt hoe "$@")
eval set -- "$cmdargs"

while [ "$#" -ne 0 ] ; do
    case "$1" in
        -h|--help)
            usage
            exit 1 ;;
        -o|--offscreen)
            sdl_driver=offscreen
            shift 1
            ;;
        -e|--egg)
            copy_egg=yes
            shift 1
            ;;
        --)
            break ;;
        *) logerror "Argument parse error, could not parse $1"
           exit 1 ;;
    esac
done

# If the CARLA egg file does not exist, copy it from the docker image
if [ -n "$copy_egg" ]; then
    echo "CARLA egg file requested, copying from \"$carla_image\" image."
    egg_file=$(docker run --rm "$carla_image" bash -c "ls ${carla_egg_dir}/*py3*.egg")
    echo "Egg file $egg_file found, copying..."
    docker cp "$(docker create --rm $carla_image)":"${egg_file}" "$script_dir"
fi

echo "Running $carla_image"
docker run \
       -p 2000-2002:2000-2002 \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY="$DISPLAY" \
       -e SDL_VIDEODRIVER="$sdl_driver"\
       -it \
       --rm \
       $nvidia_flag \
       $carla_image \
       ./CarlaUE4.sh -opengl
