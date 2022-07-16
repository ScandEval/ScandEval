"""Extension of Hugging Face TrainingArguments which supports Apple MPS GPU."""

import os

import torch
import torch.distributed as dist
from transformers import TrainingArguments
from transformers.deepspeed import is_deepspeed_available
from transformers.training_args import get_int_from_env
from transformers.utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    torch_required,
)


class TrainingArgumentsWithMPSSupport(TrainingArguments):
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":

        # Set the type of `local_rank`
        self.local_rank: int

        # If no CUDA has been requested then skip it
        if self.no_cuda:

            # Set the device to the CPU
            device = torch.device("cpu")
            self._n_gpu = 0

            # Initialise the local rank, related to distributed training
            env_keys = [
                "LOCAL_RANK",
                "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "MV2_COMM_WORLD_LOCAL_RANK",
            ]
            self.local_rank = get_int_from_env(
                env_keys=env_keys,
                default=self.local_rank,
            )

            # Initialises distributed backend for cpu, if the local rank is
            # non-negative
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                if (
                    self.xpu_backend == "ccl"
                    and int(os.environ.get("CCL_WORKER_COUNT", 0)) < 1
                ):
                    raise ValueError(
                        "CPU distributed training backend is ccl. but "
                        "CCL_WORKER_COUNT is not correctly set. Please use like "
                        "'export CCL_WORKER_COUNT = 1' to set."
                    )

                # Try to get launch configuration from environment variables set by MPI
                # launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank_env_keys = [
                    "RANK",
                    "PMI_RANK",
                    "OMPI_COMM_WORLD_RANK",
                    "MV2_COMM_WORLD_RANK",
                ]
                rank = get_int_from_env(env_keys=rank_env_keys, default=0)
                size_env_keys = [
                    "WORLD_SIZE",
                    "PMI_SIZE",
                    "OMPI_COMM_WORLD_SIZE",
                    "MV2_COMM_WORLD_SIZE",
                ]
                size = get_int_from_env(env_keys=size_env_keys, default=1)
                local_size_env_keys = [
                    "MPI_LOCALNRANKS",
                    "OMPI_COMM_WORLD_LOCAL_SIZE",
                    "MV2_COMM_WORLD_LOCAL_SIZE",
                ]
                local_size = get_int_from_env(env_keys=local_size_env_keys, default=1)

                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size or self.xpu_backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env "
                            "not set, please try exporting rank 0's hostname as "
                            "MASTER_ADDR"
                        )
                torch.distributed.init_process_group(
                    backend=self.xpu_backend, rank=rank, world_size=size
                )

        # Otherwise, check if TPU is available, and use that if that's the case
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            self._n_gpu = 0

        # Otherwise, we check if Sagemaker is available, and use that if that's the
        # case
        elif is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp

            smp.init()
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            dist.init_process_group(backend="smddp")
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK", -1))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        # Otherwise, if Deepspeed is available, we use that
        elif self.deepspeed:
            if not is_deepspeed_available():
                raise ImportError(
                    "--deepspeed requires deepspeed: `pip install deepspeed`."
                )
            import deepspeed

            deepspeed.init_distributed()

            # Workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed
            # if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        # Otherwise, if we are not using any distributed training, then we set the
        # device to be CUDA, MPS or CPU, according to availability
        elif self.local_rank == -1:

            # If CUDA is available, we use it
            if torch.cuda.is_available():
                device_name = "cuda:0"
                self._n_gpu = torch.cuda.device_count()

            # Otherwise, if MPS is available, we use it
            elif torch.backends.mps.is_available():
                device_name = "mps"
                self._n_gpu = 1

            # Otherwise, we use the CPU
            else:
                device_name = "cpu"
                self._n_gpu = 0

            # Set the device to the device name
            device = torch.device(device_name)

        # Otherwise, we are using distributed training, and we set the device to be
        # CUDA, with the associated local rank
        else:
            # This initialises the distributed backend which will take care of
            # synchronizing nodes/GPUs
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        # If we are using CUDA then we explicitly tell PyTorch to use the specified
        # device
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # Return the device
        return device
