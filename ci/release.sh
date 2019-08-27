git checkout develop
git pull
git checkout master

echo "Current version is: "
git describe --abbrev=0

# version="USER INPUT"
read -p "Enter version number to release: " version

git flow release start $version
git flow release publish $version
git flow release finish $version
git push --follow-tags