#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

echo "--- Starting Buildkite Bootstrap"

# Since Buildkite prepends uploaded steps (inserts in reverse order),
# uploading the test steps first and build steps last ensures they
# are displayed and queued in correct order: Build -> Tests.

echo "Uploading Perf and Eval Pipeline"
buildkite-agent pipeline upload .buildkite/pipeline_perf.yml

echo "Uploading Unit Tests Pipeline"
buildkite-agent pipeline upload .buildkite/pipeline_tests.yml

echo "Uploading Build Pipeline"
buildkite-agent pipeline upload .buildkite/pipeline_build.yml

echo "--- Buildkite Bootstrap Finished"
