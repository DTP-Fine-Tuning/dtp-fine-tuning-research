if [ $# -ne 2 ]; then
  echo "Usage: ./git-set-me.sh \"Nama Lengkap\" email@example.com"
  exit 1
fi

git config user.name "$1"
git config user.email "$2"

echo " Git identity updated!"
git config --list --local | grep user
