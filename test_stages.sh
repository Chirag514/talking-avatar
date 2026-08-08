#!/bin/bash
# test_stages.sh
#
# Standalone per-stage testing from the command line. Run from
# path_a_pipeline/ (one level above pipeline/).
#
# Usage:
#   ./test_stages.sh voice  <script.txt> <reference_voice.wav> [--speed 1.4]
#   ./test_stages.sh ditto  <image.png> <audio.wav>
#   ./test_stages.sh restore <video.mp4>
#   ./test_stages.sh export  <video.mp4> <audio.wav> <output.mp4>
#
# NOTE: the `ditto` subcommand must be run with the `ditto` conda env
# active (or prefixed with `conda run -n ditto`) -- it imports Ditto's
# StreamSDK directly, which only exists in that environment.

set -e

STAGE_DIR="pipeline/stages"

if [ ! -d "$STAGE_DIR" ]; then
    echo "ERROR: run this from path_a_pipeline/ (expected to find $STAGE_DIR here)."
    exit 1
fi

SUBCOMMAND=$1
shift || true

case "$SUBCOMMAND" in

    voice)
        # Forwards everything to stage2_voice_gen.py's own argparse CLI --
        # --speed / --duration / --language / --output all work here.
        python "$STAGE_DIR/stage2_voice_gen.py" "$@"
        ;;

    ditto)
        # <reference_image> <audio_wav_path> [output_path]
        # Run this with the ditto conda env active:
        #   conda activate ditto && ./test_stages.sh ditto image.png audio.wav
        IMAGE="$1"; AUDIO="$2"; OUTPUT="${3:-test_output/ditto_test.mp4}"
        if [ -z "$AUDIO" ]; then
            echo "Usage: ./test_stages.sh ditto <image.png> <audio.wav> [output.mp4]"
            exit 1
        fi
        python "$STAGE_DIR/stage4c_ditto.py" "$IMAGE" "$AUDIO" "$OUTPUT"
        ;;

    restore)
        VIDEO="$1"
        if [ -z "$VIDEO" ]; then
            echo "Usage: ./test_stages.sh restore <video.mp4>"
            exit 1
        fi
        python -c "
from pathlib import Path
import sys
sys.path.insert(0, '$STAGE_DIR/..')
from stages import stage5_restoration
result = stage5_restoration.run_restoration(Path('$VIDEO'), Path('test_output/restored.mp4'))
print(result)
"
        ;;

    export)
        VIDEO="$1"; AUDIO="$2"; OUTPUT="$3"
        if [ -z "$OUTPUT" ]; then
            echo "Usage: ./test_stages.sh export <video.mp4> <audio.wav> <output.mp4>"
            exit 1
        fi
        python -c "
from pathlib import Path
import sys
sys.path.insert(0, '$STAGE_DIR/..')
from stages import stage6_export
params = stage6_export.export_final(Path('$VIDEO'), Path('$AUDIO'), Path('$OUTPUT'))
print(params)
"
        ;;

    *)
        echo "Usage: ./test_stages.sh {voice|ditto|restore|export} [args...]"
        echo ""
        echo "Examples:"
        echo "  ./test_stages.sh voice script.txt reference_voice.wav --speed 1.4"
        echo "  ./test_stages.sh ditto photo.png test_output/voice.wav"
        echo "  ./test_stages.sh restore test_output/ditto_test.mp4"
        echo "  ./test_stages.sh export test_output/restored.mp4 test_output/voice.wav final.mp4"
        exit 1
        ;;
esac
