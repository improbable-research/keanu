#!/usr/local/bin/python3

import argparse
import fileinput
import os
from shutil import copyfile

def updateFile(destination, version):
    versiondots = version.replace('_', '.')
    with fileinput.FileInput(destination, inplace=True) as file:
        for line in file: 
            print(line.replace("permalink:", "version: " + versiondots + "\npermalink:"), end='')
    with fileinput.FileInput(destination, inplace=True) as file:
        for line in file:
            print(line.replace("/docs", "/docs/" + version), end='')

def performFreeze(version):
    directory = "current_docs"
    version = version.replace(".", "_")
    for file in os.listdir(directory):
        if not os.path.isdir(file):
            path = "current_docs/"
            legacy_path = "legacy_docs/"
            file_src = path + file
            dest_dir = legacy_path + version + "/"
            destination = dest_dir + file
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            if not os.path.exists(destination):
                open(destination, "w").close()
            copyfile(file_src, destination)
            updateFile(destination, version)


parser = argparse.ArgumentParser()
parser.add_argument("--version")
args = parser.parse_args()
performFreeze(args.version)
print("Freezing has now been performed - please make sure to update the _data/previous_versions.yml file to reflect these changes and then check in to source control.")
