echo "Current version is: "
git describe --abbrev=0

#  version="USER INPUT"
read -p "Enter version number to release: " version

git checkout master
git flow release start $version
git flow release publish $version
git flow release finish $version
git push --follow-tags
