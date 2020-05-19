#!/bin/bash

for grp in $GROUPS_IDS; do
    addgroup --gid "$grp" user_"$grp"
done

adduser --disabled-password --gecos '' --uid "$USER_ID" --gid "$GROUP_ID" user

export HOME=/home/user

cp -Rv /home/root/. $HOME/
chown -R user $HOME

GIDS=$(echo $GROUPS_IDS | sed 's/ /,/g')
usermod -aG "$GIDS" user
usermod -aG docker user

/usr/local/bin/gosu user echo "$AUTH_TOKEN" | /usr/local/bin/gosu user docker login -u oauth2accesstoken --password-stdin http://eu.gcr.io

# Toolshare v2 setup https://github.com/improbable/platform/blob/master/docs/product-groups/dev-workflow/toolshare.v2/setup-ci.md
# Running as root results in user not being able to write binaries to the directory (permissions) , so run as user
export PATH="$HOME/.improbable/imp-tool/subscriptions:$PATH"

/usr/local/bin/gosu user imp-tool subscribe --tools=imp-ci "${IMP_TOOL_FLAGS[@]}" --use_gcs_oidc_auth=false

exec /usr/local/bin/gosu user "$@"
