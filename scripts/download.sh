#!/bin/bash

function download_url() {
    local source_url=$1
    local target_dir=$2
    local target_path=$target_dir/${source_url##*/}

    echo downloading $source_url to $target_path...

    if [ -f $target_path ]; then
        echo $target_path
        echo $target_path already exits, skipping...
        return
    fi

    wget $source_url -P $target_dir/
}

function download_all_urls() {
    local url_list_file=$1
    local target_dir=$2
    local url_list=()
    readarray -t url_list < $url_list_file
    for url in ${url_list[@]}
    do
        download_url $url $target_dir
    done
}
