#!/bin/bash

############################################
# Example preparation script for switchboard
############################################
# Kaldi utility `feat-to-len` and `copy-feats` should be in your $PATH

# List split names here. These correspond to Kaldi prepared subfolder
# names as well.
splits=( train_nodup train_dev eval2000_test )

# The root folder containing the split subfolders
input_folder=~/data/swbd

# The required hierarchy is:
# ${input_folder}/split_name/
#     - utt2spk
#     - text
#     - feats.scp
#     - cmvn.scp

# Where to put the prepared nmtpy-ready files
output_folder=/tmp/data/swbd

# Create the folder
mkdir -p $output_folder

####################################################################
# REQUIREMENT CHECK
# Make sure that the files are ordered (in sync) w.r.t utterance IDs
####################################################################
for split in "${splits[@]}"; do
  # Original .scp with valid paths to .ark files such as the following
  #  sw02054-A_000204-000790 /path/to/ark/file:offset
  scp=${input_folder}/${split}/feats.scp

  # Transcription per line prefixed with utterance IDs as well
  # sw02054-A_000204-000790 so let me tell you a little bit ...
  txt=${input_folder}/${split}/text

  # NOTE: Make sure that the files are ordered (in sync) w.r.t utterance IDs
  # Compare utterance IDs to make sure that they're ordered/aligned
  cmp -s <(cut -d' ' -f1 < $scp) <(cut -d' ' -f1 < $txt) || \
    { echo "Error: [$split] feats.scp and text are not aligned"; exit 1; }
done

###############################
# Generate `segments.len` files
###############################
for split in "${splits[@]}"; do
  mkdir -p $output_folder/${split}
  scp=${input_folder}/${split}/feats.scp
  utt2spk="${input_folder}/${split}/utt2spk"
  cmvn="${input_folder}/${split}/cmvn.scp"
  scp="${input_folder}/${split}/feats.scp"
  seg=${output_folder}/${split}/segments.len

  if [[ ! -f $seg ]]; then
    # Extract frame counts
    echo "Extracting frame counts for $split"
    feat-to-len scp:$scp ark,t:- | cut -d' ' -f2 > ${output_folder}/${split}/segments.len
  fi

  if [[ ! -f "${output_folder}/${split}/feats_local.ark" ]]; then
    feats_cmvn="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$utt2spk scp:$cmvn scp:$scp ark:- |"
    copy-feats "$feats_cmvn" ark,scp:`realpath $output_folder/${split}/feats_local.ark`,$output_folder/${split}/feats_local.scp &
  fi
done

# Wait for completion
wait
