import os

try:
    os.rename('_build/html/_static', '_build/html/static')
    os.rename('_build/html/_modules', '_build/html/modules')
    os.rename('_build/html/_sources', '_build/html/sources')
except:
    pass

root_dir = '_build/html'

for directory, subdirectories, files in os.walk(root_dir):
    for fileName in files:
        try:
            fileName = os.path.join(directory, fileName)
            file = open(fileName, 'r')
            contents = file.read()
            file.close()
            file = open(fileName, 'w')
            replaced_contents = contents.replace('_static', 'static')
            replaced_contents = replaced_contents.replace('_modules', 'modules')
            replaced_contents = replaced_contents.replace('_sources', 'sources')
            file.write(replaced_contents)
        except:
            pass

print("Finished renaming all directories and mentions of directories with underscores")