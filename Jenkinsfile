pipeline {
    agent {
        dockerfile true
    }
    environment {
        /* There are systems where shebang lines cannot be longer than
        127 characters, which may be an issue with the long paths on continuous
        integration servers. In our case, this happens for tox with generated
        scripts such as `.tox/py27/bin/pip`, with an error like
        "OSError: [Errno 2] No such file or directory".
        We work around this by changing the tox working folder to a temporary
        folder, as this has a shorter path. See also
        http://tox.readthedocs.io/en/latest/example/jenkins.html. */
        TOXWORKDIR = sh (
            script: 'mktemp --directory --suffix .tox',
            returnStdout: true
        ).trim()
    }
    stages {
        stage('Build') {
            steps {
                sh 'ci/build.sh'
            }
        }
    }
}