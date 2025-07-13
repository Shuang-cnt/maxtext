# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""User-configurable settings for running MaxText benchmarks.

This file defines the `UserConfig` dataclass, which serves as a centralized
location for users to specify their GCP environment, desired models, and
execution parameters for running benchmarks with XPK. The `USER_CONFIG`
instance at the bottom of the file is the main object to be modified for
custom runs.
"""

import dataclasses
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from .. import maxtext_trillium_model_configs as v6e_model_configs
from .. import maxtext_v5e_model_configs as v5e_model_configs
from .. import maxtext_v5p_model_configs as v5p_model_configs
from .pw_utils import build_user_models, get_cluster_config, get_pathways_config


AVAILABLE_MODELS_FRAMEWORKS = ["mcjax", "pathways"]

AVAILABLE_MODELS = {
    "v6e": v6e_model_configs.trillium_model_dict,
    "v5litepod": v5e_model_configs.v5e_model_dict,
    "v5p": v5p_model_configs.v5p_model_dict,
}


@dataclasses.dataclass
class UserConfig:
  """The default configuration can be modified here."""

  # gcp configuration
  user: str = "user_name"
  cluster_name: str = "pw-scale-test-v5e-32"
  project: str = "cloud-tpu-multipod-dev"
  zone: str = "us-south1"
  device_type: str = "v5litepod-32"
  priority: str = "medium"

  # Images for env
  server_image: str = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server_maxtext:latest"
  proxy_image: str = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server_maxtext:latest"
  runner: str = "gcr.io/cloud-tpu-multipod-dev/sujinesh_maxtext_latest"
  colocated_python_image: str = "gcr.io/cloud-tpu-multipod-dev/ksadi_sidecar_maxtext:latest"
  worker_flags: str = ""
  proxy_flags: str = ""
  server_flags: str = ""

  # model configuration
  benchmark_steps: int = 25
  headless: bool = True
  selected_model_framework: list[str] = dataclasses.field(default_factory=lambda: ["pathways"])  # pathways, mcjax
  selected_model_names: list[str] = dataclasses.field(default_factory=lambda: ["llama3_1_8b_8192_v5e_32"])
  num_slices_list: list[int] = dataclasses.field(default_factory=lambda: [1])

  # other configuration
  xpk_path: str = "~/xpk"
  max_restarts: int = 0

  def __post_init__(self):
    """Automatically generate derived attributes after the object is created."""
    self.cluster_config = get_cluster_config(self.cluster_name, self.project, self.zone, self.device_type)

    self.region = "-".join(self.zone.split("-")[:-1])
    self.pathways_config = get_pathways_config(
        self.server_image,
        self.proxy_image,
        self.runner,
        self.colocated_python_image,
        self.headless,
        self.server_flags,
        self.proxy_flags,
        self.worker_flags,
    )
    self.headless_workload_name = f"{self.user[:3]}-headless"
    self.base_output_directory = f"gs://{self.user}-{self.region}/{self.user}-"

    device_base_type = self.device_type.split("-", maxsplit=1)[0]
    self.models = build_user_models(
        self.selected_model_framework,
        self.selected_model_names,
        device_base_type,
        AVAILABLE_MODELS_FRAMEWORKS,
        AVAILABLE_MODELS,
    )


# Define the required configuration here
USER_CONFIG = UserConfig(
    user=os.environ.get("USER", "default_user"),
    cluster_name="pw-scale-test-v5e-32",
    project="cloud-tpu-multipod-dev",
    zone="us-south1",
    device_type="v5litepod-32",
    benchmark_steps=25,
    num_slices_list=[1],
    server_image="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server_maxtext:latest",
    proxy_image="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server_maxtext:latest",
    runner="gcr.io/cloud-tpu-multipod-dev/sujinesh_maxtext_latest",
    colocated_python_image="gcr.io/cloud-tpu-multipod-dev/ksadi_sidecar_maxtext:latest",
    selected_model_framework=["pathways"],
    selected_model_names=["llama3_1_8b_8192_v5e_32"],
    priority="medium",
    headless=True,
)

