#!/bin/bash

pdfdir=''

while getopts 'd:' flag; do
    case "$flag" in
        d) pdfdir=$OPTARG ;;
    esac
done

for pdf in $(ls ${pdfdir}/img*.pdf); do
    fname="${pdf%.*}"
    convert -density 300x300 -quality 95 $pdf ${fname}.jpg
done
