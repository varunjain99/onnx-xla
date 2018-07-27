#!/bin/bash

# Don't source setup.sh here, because the virtualenv might not be set up yet

export NUMCORES=`grep -c ^processor /proc/cpuinfo`
if [ ! -n "$NUMCORES" ]; then
  export NUMCORES=`sysctl -n hw.ncpu`
fi
echo Using $NUMCORES cores

# Install dependencies
sudo apt-get update
APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends'
$APT_INSTALL_CMD dos2unix

function install_protobuf() {
    # Install protobuf
    local pb_dir="$HOME/.cache/pb"
    mkdir -p "$pb_dir"
    wget -qO- "https://github.com/google/protobuf/releases/download/v${PB_VERSION}/protobuf-${PB_VERSION}.tar.gz" | tar -xz -C "$pb_dir" --strip-components 1
    ccache -z
    cd "$pb_dir" && ./configure && make -j${NUMCORES} && make check && sudo make install && sudo ldconfig && cd -
    ccache -s
}

install_protobuf

sudo add-apt-repository ppa:openjdk-r/ppa:ubuntu-x-swat/x-updates
sudo apt-get update && sudo apt-get -y install openjdk-8-jdk

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get -y install bazel

# Update all existing python packages
pip install -U pip setuptools

pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

