#  version="USER INPUT"
read -p "Enter version number: " version

git checkout master
git flow release start $version
git flow release publish $version
git flow release finish $version
git push
git push origin --tags