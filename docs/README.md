# keanu-docs
```
    __ __                      
   / //_/__  ____ _____  __  __
  / ,< / _ \/ __ `/ __ \/ / / /
 / /| /  __/ /_/ / / / / /_/ / 
/_/ |_\___/\__,_/_/ /_/\__,_/  
        documentation                  
```
by Improbable

The source for the online documentation.

[Internal documentation here](http://brevi.link/keanu)

## Installation 
These docs use [Jekyll](https://jekyllrb.com/docs/). Here are the steps you need to follow to get it up and running locally.

If you don't have Ruby installed:
* if you're using Windows, use [RubyInstaller](https://rubyinstaller.org/downloads/) to get Ruby with Devkit set up
* if you're using Mac, run `brew install ruby` from a terminal window.

Install Jekyll.
```bash
gem install jekyll bundler
```
Move to _keanu-docs_ project directory and install project dependencies.
```bash
cd <keanu-docs_project_dir>
bundle install
```
Optional: you may have issues with installing nokogiri. We solved our issues using the following command
```bash
gem install nokogiri -- --use-system-libraries --with-xml2-include=/usr/include/libxml2 --with-xml2-lib=/usr/lib
```
Now you can use our python script to generate the docs with snippets included...
```bash
python bin/snippet_writer.py
```
...and finally a local server that allows you to view the webpage at `localhost:4000`
```bash
bundle exec jekyll serve
```

## Creating legacy docs
When Keanu is updated to a new version, we want to be able to freeze the documentation and save it for future reference so people can see docs for previous Keanu versions. 
If you run `python bin/freezeAtVersion.py --version xx.xx.xx` then this will generate some legacy docs from the current docs and place them in a folder `legacy_docs/xx.xx.xx/` which can be accessed by the url `docs/xx.xx.xx/docspage`.
**You need to commit these to source control**.
Then all you need to do, is update the file `_data/previous_versions.yml` such that it has an updated link to the current version of keanu, and an entry for the link you will have just created. **You also need to commit this to source control**.
See the file for an example entry.