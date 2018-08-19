#! /usr/bin/env bash
set -eu

TARGET="$(dirname $(realpath $BASH_SOURCE))/src/lib.rs"

cat - << EOD > "$TARGET"
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

EOD

cd $(mktemp -d)
echo '#include <menoh/menoh.h>' > wrapper.h
bindgen wrapper.h \
        --no-derive-copy \
        --no-derive-debug \
        --no-doc-comments \
        --no-prepend-enum-name \
        --no-recursive-whitelist \
        --whitelist-function 'menoh_.*' \
        --whitelist-type 'menoh_.*' \
        >> "$TARGET"
