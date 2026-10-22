#!/bin/sh
# Every hardware-free check of final/: ruff over the tracked Python on the host, then every
# test_* function of every tracked test_*.py in the container. Run from anywhere; the
# pre-commit hook runs it whenever final/ is staged. The vendor agx_reference is linted
# only by the edit hook, as in workspace/; its test of our feed_forward.py runs here.
set -eu
cd "$(dirname "$0")"

py=$(git ls-files 'src/*.py' | grep -v '^src/agx_reference/')
ruff check -q $py
ruff format --check -q $py

# A module name per test file, on final/src's two import roots (run.sh puts both on the path).
modules=$(git ls-files 'src/*test_*.py' | sed -e 's|^src/agx_reference/||' -e 's|^src/||' -e 's|\.py$||' -e 's|/|.|g' | tr '\n' ' ')
docker exec piper-ergodic bash -lc "cd /home/jens/workspace/docker_intern_PiPER_ergodic/final/src && sh run.sh -c '
import importlib
for name in \"$modules\".split():
    module = importlib.import_module(name)
    for test, f in sorted(vars(module).items()):
        if test.startswith(\"test_\"):
            f()
            print(\"ok \", name, test)
'"
