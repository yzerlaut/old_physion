#!/bin/bash
script="move_files"
#Declare the number of mandatory args
margs=2

# Common functions - BEGIN
function example {
    echo -e "example: $script -s VAL -d VAL"
}

function usage {
    echo -e "usage: $script MANDATORY [OPTION]\n"
}

function help {
    usage
    echo -e "MANDATORY:"
    echo -e "  -s, --source  VAL  The source location:"
    echo -e "                          -> can be: laptop, desktop_drive, desktop  "
    echo -e "  -d, --destination  VAL  The destination computer"
    echo -e "                          -> can be: laptop, desktop_drive, desktop  "
    echo -e "OPTION:"
    echo -e "  --raw_Calcium                   the raw Calcium signals"
    echo -e " -pc,  --processed_Calcium        the processed Calcium signals"
    echo -e "  --raw_FaceCamera                the raw FaceCamera movies"
    echo -e " -pf,  --processed_FaceCamera     the processed Calcium signals"
    echo -e " -a,  --all                       [MOVE ALL]"
    echo -e "  -h,  --help             Prints this help\n"
    example
}

# Ensures that the number of passed args are at least equals
# to the declared number of mandatory args.
# It also handles the special case of the -h or --help arg.
function margs_precheck {
    if [ $2 ] && [ $1 -lt $margs ]; then
	if [ $2 == "--help" ] || [ $2 == "-h" ]; then
	    help
	    exit
	else
	    usage
	    example
	    exit 1 # error
	fi
    fi
}

# Ensures that all the mandatory args are not empty
function margs_check {
    if [ $# -lt $margs ]; then
	usage
	example
	exit 1 # error
    fi
}
# Common functions - END

# Custom functions - BEGIN
# Put here your custom functions
# Custom functions - END

# Main
margs_precheck $# $1

source='desktop_drive'
destination='laptop'
source_path='/media/yann/DATATDRIVE1/'
destination_path='/home/yann/DATA/'

# Args while-loop
while [ "$1" != "" ];
do
    case $1 in
	-d  | --destination )  shift
			       destination=$1
			       ;;
	-s  | --source )  shift
			       source=$1
			       ;;
	-dp  | --destination_path )  shift
			       destination_path=$1
			       ;;
	-sp  | --source_path )  shift
			       source_path=$1
			       ;;
	-o0  | --optional0  )  oarg0="true"
			       ;;
	-o1  | --optional1  )  shift
			       oarg1=$1
			       ;;
	-h   | --help )        help
			       exit
			       ;;
	*)
	    echo "$script: illegal option $1"
	    usage
	    example
	    exit 1 # error
	    ;;
    esac
    shift
done

# Pass here your mandatory args for check
margs_check $source $destination $source_path $destination_path

# Your stuff goes here
echo $source $destination $source_path $destination_path
