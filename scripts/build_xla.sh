#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname "$script_path"))
pt_dir="$top_dir/third_party/pytorch"
tf_dir="$pt_dir/third_party/tensorflow"
build_dir="$top_dir/build/xla"
mkdir -p "$build_dir"

cd "$tf_dir"
git reset --hard
git clean -f
patch -p1 < $pt_dir/tensorflow.patch

# TODO: upstream this
patch -p1 <<EOF
diff --git a/tensorflow/workspace.bzl b/tensorflow/workspace.bzl
index 3935992..00216df 100644
--- a/tensorflow/workspace.bzl
+++ b/tensorflow/workspace.bzl
@@ -200,6 +200,7 @@ def tf_workspace(path_prefix="", tf_repo_name=""):
       urls = [
           "https://mirror.bazel.build/www.nasm.us/pub/nasm/releasebuilds/2.12.02/nasm-2.12.02.tar.bz2",
           "http://pkgs.fedoraproject.org/repo/pkgs/nasm/nasm-2.12.02.tar.bz2/d15843c3fb7db39af80571ee27ec6fad/nasm-2.12.02.tar.bz2",
+          "http://www.nasm.us/pub/nasm/releasebuilds/2.12.02/nasm-2.12.02.tar.bz2",
       ],
       sha256 = "00b0891c678c065446ca59bcee64719d0096d54d6886e6e472aeee2e170ae324",
       strip_prefix = "nasm-2.12.02",
EOF

bazel build -c opt //tensorflow/compiler/tf2xla/lib:util
bazel build -c opt //tensorflow/compiler/xla/rpc:libxla_computation_client.so
bazel build -c opt //tensorflow/compiler/xla/rpc:grpc_service_main_cpu
