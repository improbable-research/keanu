#!/usr/local/bin/python3

import argparse
import fileinput
import os
from shutil import copytree

def update_file(destination, version):
    versiondots = version.replace('_', '.')
    with fileinput.FileInput(destination, inplace=True) as file:
        for line in file: 
            print(line.replace("permalink:", "version: " + versiondots + "\npermalink:"), end='')
    with fileinput.FileInput(destination, inplace=True) as file:
        for line in file:
            print(line.replace("/docs", "/docs/" + version), end='')

def freeze_shiny(version):
    dest_dir = copy_to_versioned_legacy_path("current_docs/", "legacy_docs/", version)
    for file in os.listdir(dest_dir):
        if not os.path.isdir(file):
            update_file(dest_dir + file, version)

def copy_to_versioned_legacy_path(source_path, legacy_root, version):
    dest_dir = legacy_root + version + "/"
    copytree(source_path, dest_dir)
    return dest_dir

def freeze_python(version):
    copy_to_versioned_legacy_path("python/latest/", "python/", version)

parser = argparse.ArgumentParser()
parser.add_argument("--version")
args = parser.parse_args()
formatted_version = args.version.replace(".", "_")
freeze_shiny(formatted_version)
freeze_python(formatted_version)
print("Freezing has now been performed - please make sure to update the _data/previous_versions.yml file to reflect these changes and then check in to source control.")
