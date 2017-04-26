#!/bin/bash
wget -nc "http://downloads.tatoeba.org/exports/sentences.tar.bz2"
wget -nc "http://downloads.tatoeba.org/exports/links.tar.bz2"
tar xjvfk sentences.tar.bz2
tar xjvfk links.tar.bz2

LEN=`wc -l sentences.csv`
echo "Downloaded $LEN sentences"
