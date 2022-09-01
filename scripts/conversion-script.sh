#!/bin/bash

# Usage: bash script.sh the_main_dataset_folder_path/

rm -rf ./jpgfiles/
mkdir ./jpgfiles/

LIST=$(find $1 | grep nii.gz)

for i in $LIST
do
  OUTPUT_FILE=$(basename $i .nii.gz)
  med2image -i $i -d ./jpgfiles/ -o $OUTPUT_FILE --outputFileType raw_files
done
