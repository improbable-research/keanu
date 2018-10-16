#!/bin/bash

if [[ $TRAVIS_BRANCH == 'master' ]]; then
  ./gradlew clean build uploadArchives -PnexusUser=$NEXUS_USER -PnexusPassword=$NEXUS_PASSWORD -Psigning.keyId=$SIGNING_KEY_ID -Psigning.password=$SIGNING_PASSWORD -Psigning.secretKeyRingFile=../deployment/secret-keys-keanu.gpg --info --stacktrace
else
  ./gradlew check
fi

if [[ ! -z $SONAR_TOKEN ]]; then
  sonar-scanner
fi