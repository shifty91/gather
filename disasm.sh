#!/usr/bin/env bash
#
# Copyright (C) 2015 Kurt Kanzenbach <kurt@kmk-computers.de>
#
# This script compiles gather with clang, gcc, icc and dumps the disassembly.
#

set -e

COMPILER=""
VEC="sse2 sse4.1 avx avx2 avx512f"

[ -x `which g++     2>/dev/null` ] && COMPILER="$COMPILER g++"
[ -x `which clang++ 2>/dev/null` ] && COMPILER="$COMPILER clang++"
[ -x `which icpc    2>/dev/null` ] && COMPILER="$COMPILER icpc"
[ -d "dumps" ] || mkdir -p "dumps"

for compiler in $COMPILER ; do
  for vec in $VEC ; do
    make clean
    make ADTFLAGS=-m${vec} CXX=$compiler PROG=gather_${vec}_${compiler}
    objdump -d -Mintel gather_${vec}_${compiler} > dumps/gather_${vec}_${compiler}
  done
done

exit 0
