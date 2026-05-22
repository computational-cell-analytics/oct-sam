#!/bin/bash
#
# Author:
# Martin Schilling, 2026, martin.schilling@med.uni-goettingen.de
#
# Process OCT images with a pre-trained nnU-Net for the segmentation of retinal layers.

SCRIPT_DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"

helpstr=$(cat <<- EOF
Process images in an input directory using nnU-Net.

-n nnunet env   Name of environment for nnU-Net
-o oct-sam env  Name of environment for OCT-SAM
-h help
EOF
)

ENV_OCT_SAM="oct-sam"
ENV_NNUNET="nnunet"

usage="Usage: $0 [-h] [-n nnunet env] [-o oct-sam env] <input_dir> <output_dir>"

while getopts "n:o:h" opt; do
    case $opt in
    n)
        ENV_NNUNET=("$OPTARG")
    ;;
    o)
        ENV_OCT_SAM=("$OPTARG")
    ;;
    h)
        echo "$usage"
        echo
        echo "$helpstr"
        exit 0
    ;;
    \?)
        echo "$usage" >&2
        exit 1
    ;;
    esac
done

shift $((OPTIND - 1))

if [ $# -lt 2 ] ; then

    echo "$usage" >&2
    exit 1
fi

INPUT=$(readlink -f "$1")
OUTPUT=$(readlink -f "$2")

WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR

NIFTI_IMAGES="$WORKDIR"/images
NIFTI_SEGMENTATIONS="$WORKDIR"/segmentations
mkdir "$NIFTI_IMAGES"
mkdir "$NIFTI_SEGMENTATIONS"

# Converting image data into nnU-Net data format
micromamba run -n "$ENV_OCT_SAM" python "$SCRIPT_DIR"/process_nnunet_data/convert_to_inference_data.py -i "$INPUT" -o "$NIFTI_IMAGES"

# Applying nnU-Net
micromamba run -n "$ENV_NNUNET" nnUNetv2_predict -i "$NIFTI_IMAGES" -o "$NIFTI_SEGMENTATIONS" -d 001 -c 2d -f 0 -device cpu

# Convert output format of nnU-Net back to TIF
micromamba run -n "$ENV_OCT_SAM" python "$SCRIPT_DIR"/process_nnunet_data/convert_nifti_to_tif.py -i "$NIFTI_SEGMENTATIONS" -o "$OUTPUT" --label

echo "Output folder:" "$OUTPUT"
