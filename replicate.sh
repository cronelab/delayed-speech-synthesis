#! /bin/bash

# Specify data directories
DATA_DIR=/.../.../KeywordReading
NORM_DIR=/.../.../SyllableRepetition
LIVE_DIR=/.../.../KeywordReading/online_sessions
TEMP_DIR=/tmp/ReplicateDelayedSynthesis

settings=config/debug_settings.ini

contamination_package_path=".../Contamination Analysis Package/Toolbox"
stage=1
stop_stage=8

# -------------------------------------------------------------------------------------------------------
# CONTAMINATION ANALYSIS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Running the contamination analysis part"

  # Check if matlab command is available
  if ! command -v matlab &> /dev/null; then echo "Matlab command is not available."; exit 1; fi

  mkdir -p $TEMP_DIR/contamination
  env PYTHONPATH=./eval/contamination:$PYTHONPATH python eval/contamination/aggregate_per_day.py  \
    --corpus-root $DATA_DIR                                                                       \
    --acc-path $TEMP_DIR/contamination/aggregated_by_day                                          \
    --timing-path $TEMP_DIR/contamination/timings

  mkdir -p $TEMP_DIR/contamination/prepared
  matlab -nodesktop -nosplash -r "addpath(genpath('eval/contamination')); \
    data_preparation('$TEMP_DIR/contamination/prepared', \
    '$TEMP_DIR/contamination/aggregated_by_day', \
    '$contamination_package_path'); exit;"

  mkdir -p $TEMP_DIR/contamination/analysis
  matlab -nodesktop -nosplash -r "addpath(genpath('eval/contamination')); \
    run_contamination_analysis('$TEMP_DIR/contamination/analysis', \
    '$TEMP_DIR/contamination/prepared', \
    '$TEMP_DIR/contamination/timings', \
    '$contamination_package_path'); exit;"

  mkdir -p $TEMP_DIR/analysis
  env PYTHONPATH=./eval/contamination:$PYTHONPATH python eval/contamination/gen_contamination_report.py  \
    $TEMP_DIR/contamination                                                                              \
    --out $TEMP_DIR/analysis
fi

# -------------------------------------------------------------------------------------------------------
# PREPARE DATA & COMPUTE HIGH-GAMMA FEATURES
# -------------------------------------------------------------------------------------------------------
corpus_dir=$TEMP_DIR/corpus
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Precompute features to for train, validation and test set"
    mkdir -p $corpus_dir
    python prepare_corpus.py $corpus_dir $NORM_DIR $DATA_DIR
fi

# -------------------------------------------------------------------------------------------------------
# TRAIN THE UNIDIRECTIONAL VAD MODEL
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Train nVAD model"
    python train_unidirectional_vad.py $corpus_dir $TEMP_DIR/nVAD  \
      --test_day 2022_11_03                                        \
      --val_day 2022_11_04                                         \
      --epochs 8
fi

# -------------------------------------------------------------------------------------------------------
# TRAIN THE BIDIRECTIONAL DECODING MODEL
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Train bidirectional decoding model"
    python train_bidirectional_model.py $corpus_dir $TEMP_DIR/decoding_model  \
      --test_day 2022_11_03                                                   \
      --val_day 2022_11_04                                                    \
      --epochs 20
fi

# -------------------------------------------------------------------------------------------------------
# COMPUTE DAY SPECIFIC NORMALIZATION STATISTICS
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Compute day specific normalization statistics"
    python baseline_offline.py $settings
fi

# -------------------------------------------------------------------------------------------------------
# RENDER POWER SPECTRAL ANALYSIS PLOT
# -------------------------------------------------------------------------------------------------------
corpus_dir=$TEMP_DIR/corpus
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: Generate supplementary figure 2"
    mkdir -p $TEMP_DIR/analysis
    env PYTHONPATH=./eval:$PYTHONPATH python eval/suppl_fig_2.py  \
      $DATA_DIR/2022_09_22/KeywordReading_Overt_R01.mat           \
      $NORM_DIR/2022_09_22/SyllableRepetition_Overt.mat           \
      --out $TEMP_DIR/analysis
fi

# -------------------------------------------------------------------------------------------------------
# STREAM DATA LOCALLY USING THE DEVELOPMENT AMPLIFIER
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Play one file from the online test days locally to synthesize speech (will run for 60 seconds)"
    python development_amplifier.py $LIVE_DIR/2023_04_14/KeywordSynthesis_Overt_R01.mat --seconds 60 &
fi

# -------------------------------------------------------------------------------------------------------
# START ONLINE DECODER IN THE BACKGROUND
# -------------------------------------------------------------------------------------------------------
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Stage 8: Starting online decoder (running in the background, close using STRG-C)"
    python decode_online.py $settings --run replicate --overwrite |  \
      play -t raw -r 16000 -e signed -b 16 -c 1 --buffer 256 -V0 -q -
fi
