#!/usr/local/bin/python3

# Copyright 2018, Improbable

"""A simple script to pull out snippets from a set of source files and write them out in to a tagged file

   Snippets are the set of lines between a %%SNIPPET_START%% <name> and %%SNIPPET_END%% <name> pair

   Snippets are inserted at any point in the input file with a corresponding {% snippet <name> %}"""


import argparse
import re
import os
import sys
import logging

error_code = 0

def record_error(output_str):
    logging.error(output_str)
    global error_code
    error_code = 1

def printd(debug_str):
    logging.debug(debug_str)

def read_file_snippets(file, snippet_store):
    """Parse a file and add all snippets to the snippet_store dictionary"""
    start_reg = re.compile("(.*%%SNIPPET_START%% )([a-zA-Z0-9]+)")
    end_reg =   re.compile("(.*%%SNIPPET_END%% )([a-zA-Z0-9]+)")
    open_snippets = {}
    with open(file) as w:
        lines = w.readlines()

        for line in lines:
            printd("Got Line: {}".format(line))
            # Check whether we're entering or leaving a snippet
            m = start_reg.match(line)
            if m:
                printd("Opened Snippet {}".format(m.group(2)))
                if m.group(2) in snippet_store:
                    record_error("Repeat definition of Snippet {}".format(m.group(2)))
                elif m.group(2) in open_snippets:
                    record_error("Snippet already opened {}".format(m.group(2)))
                else:
                    printd("Added {} to open snippets list".format(m.group(2)))
                    open_snippets[m.group(2)] = []
                continue

            m = end_reg.match(line)
            if m:
                printd("Found end of Snippet {}".format(m.group(2)))
                if m.group(2) not in open_snippets:
                    record_error("Reached Snippet End but no start")
                elif m.group(2) in snippet_store:
                    record_error("Repeat definition of Snippet {}".format(m.group(2)))
                else:
                    snippet_store[m.group(2)] = open_snippets[m.group(2)]
                    del open_snippets[m.group(2)]
                continue

            # If we've got this far, then we're just a normal line, so we can add this to all open snippets
            for snippet in open_snippets.values():
                printd("Adding Line to snippet")
                snippet.append(line)

        # Now, warn about any unclosed snippets
        for opened in open_snippets:
            record_error("Snippet {} left open - ignoring".format(opened))


def replace_tags(in_name, out_name, snippet_store):
    tag_re = re.compile("(.*{% snippet )([a-zA-Z0-9]+)( %})")
    with open(in_name) as in_file, open(out_name, "w") as out_file:
        for in_line in in_file.readlines():
            m = tag_re.match(in_line)

            # If this isn't a matching line, then just write it to the output file.  If it is, then find the appropriate
            # snippet and write that out to the output file instead
            if m:
                printd("Found tag (replacing): {}".format(m.group(2)))
                if m.group(2) in snippet_store:
                    snippet = snippet_store[m.group(2)]
                    out_file.write("".join(snippet))
                else:
                    record_error("Snippet not found: {}".format(m.group(2)))
            else:
                out_file.write(in_line)


def do_rewrites(file_name, in_loc, out_loc, snippet_store):
    printd("file name: {}\nin_loc: {}\nout_loc: {}".format(file_name, in_loc, out_loc))
    replace_tags(in_loc + file_name,
                 out_loc + file_name,
                 snippet_store)


def for_all_in_dir(directory, action):
    for file in os.listdir(directory):
        action(file)

def subtractStrings(str1, str2):
    if str2 in str1:
        return str1.replace(str2, '')
    return ""

def for_all_in_subdirs(directory, action):
    for path, subdirs, files in os.walk(directory):
        for file in files:
            path = subtractStrings(path, directory)
            action(os.path.join(path, file))

def check_output_dir_exists(output_dir):
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)


def main():
    """Parse args (expects source and dest doc directories and snippet source dir)
       read all snippets and then process the input files
       writing out the new versions to the output location"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="The Input file directory", default="doc_templates/")
    parser.add_argument("--output_dir", help="Where to store the processed files", default="current_docs/")
    parser.add_argument("--src_dir", help="Where snippet source files are located", default="src/test/java/io/improbable/snippet/")
    parser.add_argument("--debug", action='store_true', help="Turn on script debugging")

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level = logging.DEBUG)

    snippet_store = {}
    for_all_in_dir(args.src_dir, lambda x: read_file_snippets(args.src_dir + x, snippet_store))
    printd(str(snippet_store))
    check_output_dir_exists(args.output_dir)
    for_all_in_subdirs(args.input_dir, lambda x: do_rewrites(x,
                                                              args.input_dir,
                                                              args.output_dir,
                                                              snippet_store))

if __name__ == "__main__":
    main()
    sys.exit(error_code)
