set -ex

export top_dir=$(dirname ${0%/*})

# setup ccache
export PATH="/usr/lib/ccache:$PATH"
ccache --max-size 1G
