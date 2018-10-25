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
* if you're using Windows, use [RubyInstaller](https://rubyinstaller.org/downloads/) to get Ruby with Devkit set up;
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

## Compiling from a local build of Keanu
Occasionally you may wish to write examples in the documentation that are dependent on new features in Keanu that have not yet been released. 
In this situation it is useful to compile these docs using a local build of Keanu. 
To do so, you will first need to publish keanu to MavenLocal - a local repository of maven packages. 
First, go to the `build.gradle` file in the top level of the Keanu project and add the `maven-publish` plugin and the following publishing field. 

```groovy
plugins {
    ...
    id 'maven-publish'
}

...

publishing {
    publications {
        keanu(MavenPublication) {
            from components.java
        }
    }

    repositories {
        maven {
            name = 'keanu'
            url = "file://${buildDir}/repo"
        }
    }
}
```

Now you can go to Keanu directory and run `./gradlew publishToMavenLocal` which will do what it says on the tin.

The next thing you need to do is to find what version of Keanu has been compiled and published to MavenLocal.
To do this, you can explore the `~/.m2` directory by running a command line `find ~/.m2/ -name 'keanu*'`.
You will get an output like the following:

```
/Users/charliecrisp/.m2//repository/io/improbable/keanu
/Users/charliecrisp/.m2//repository/io/improbable/keanu/v0.0.13-24-ga53ffb7.dirty/keanu-v0.0.13-24-ga53ffb7.dirty.jar
/Users/charliecrisp/.m2//repository/io/improbable/keanu/v0.0.13-24-ga53ffb7.dirty/keanu-v0.0.13-24-ga53ffb7.dirty.pom
/Users/charliecrisp/.m2//repository/io/improbable/keanu/v0.0.13-24-ga53ffb7.dirty/keanu-v0.0.13-24-ga53ffb7.dirty-sources.jar
/Users/charliecrisp/.m2//repository/io/improbable/keanu/v0.0.13-24-ga53ffb7.dirty/keanu-v0.0.13-24-ga53ffb7.dirty-javadoc.jar
/Users/charliecrisp/.m2//repository/io/improbable/keanu/v0.0.13-35-gf2e4c87.dirty/keanu-v0.0.13-35-gf2e4c87.dirty.jar
/Users/charliecrisp/.m2//repository/io/improbable/keanu/v0.0.13-35-gf2e4c87.dirty/keanu-v0.0.13-35-gf2e4c87.dirty.pom
/Users/charliecrisp/.m2//repository/io/improbable/keanu-project
/Users/charliecrisp/.m2//repository/io/improbable/keanu-project/v0.0.13-35-gf2e4c87.dirty/keanu-project-v0.0.13-35-gf2e4c87.dirty.pom
/Users/charliecrisp/.m2//repository/io/improbable/keanu-project/v0.0.13-35-gf2e4c87.dirty/keanu-project-v0.0.13-35-gf2e4c87.dirty.jar
```

The key information here is that you have a bunch of publications of Keanu with a version  (e.g. `v0.0.13`), a number of commits ahead of the latest release (e.g. `24` or `35` in my case) and a commit hash (e.g. `gf2e4c87`).
Now that you have this information, you can head to `build.gradle` in the project that requires the experimental keanu as a dependency (i.e. this dcos project) and change the line that says `compile "io.improbable:keanu:+"` to `compile "io.improbable:keanu:YOUR_VERSION"` where `YOUR_VERSION` will be something like `v0.0.13-35-gf2e4c87.dirty`.
Now run `./gradlew build` and you should be using your own experimental version of Keanu.

## Creating legacy docs
When Keanu is updated to a new version, we want to be able to freeze the documentation and save it for future reference so people can see docs for previous Keanu versions. 
If you run `python bin/freezeAtVersion.py --version xx.xx.xx` then this will generate some legacy docs from the current docs and place them in a folder `legacy_docs/xx.xx.xx/` which can be accessed by the url `docs/xx.xx.xx/docspage`.
**You need to commit these to source control**.
Then all you need to do, is update the file `_data/previous_versions.yml` such that it has an updated link to the current version of keanu, and an entry for the link you will have just created. **You also need to commit this to source control**.
See the file for an example entry.