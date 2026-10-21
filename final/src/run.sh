#!/bin/sh
# Run a final/ script on final/'s own import roots: the container's PYTHONPATH names the
# sandbox's workspace/src, which final/ never imports.  Usage: sh run.sh label.py <args>
DIR=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="$DIR:$DIR/agx_reference" exec python "$@"
