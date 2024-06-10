import argparse

from dask_kubernetes.operator.kubecluster import KubeCluster, make_cluster_spec

from nemo_curator.utils.script_utils import ArgumentHelper


def create_cluster(
    name: str,
    n_workers: int,
    n_gpus_per_worker: int,
    n_cpus_per_worker: int,
    image: str,
    image_pull_secret: str,
    pvcs: dict[str, str],
):
    dask_worker_command = "dask-worker"
    if n_gpus_per_worker and n_gpus_per_worker > 0:
        dask_worker_command = "dask-cuda-worker"

    custom_cluster_spec = make_cluster_spec(
        name=name,
        worker_command=dask_worker_command,
        n_workers=n_workers,
        image=image,
    )
    scheduler_spec = custom_cluster_spec["spec"]["scheduler"]["spec"]
    worker_spec = custom_cluster_spec["spec"]["worker"]["spec"]
    if image_pull_secret:
        scheduler_spec["imagePullSecrets"] = [{"name": image_pull_secret}]
        worker_spec["imagePullSecrets"] = [{"name": image_pull_secret}]

    obj_vols = []
    obj_vol_mounts = []
    for pvc_name, mount_path in pvcs.items():
        obj_vols.append(
            {
                "name": pvc_name,
                "persistentVolumeClaim": {
                    "claimName": pvc_name,
                },
            }
        )
        obj_vol_mounts.append(
            {
                "name": pvc_name,
                "mountPath": mount_path,
            }
        )

    scheduler_spec["volumes"] = obj_vols
    for ctr in scheduler_spec["containers"]:
        ctr["volumeMounts"] = obj_vol_mounts

    worker_spec["volumes"] = obj_vols
    for ctr in worker_spec["containers"]:
        ctr["volumeMounts"] = obj_vol_mounts
        # Resources are added to only the worker, since the scheduler doesn't need GPUs
        if n_gpus_per_worker or n_cpus_per_worker:
            if not ctr["resources"]:
                ctr["resources"] = {"limits": {}}
            if n_gpus_per_worker:
                ctr["resources"]["limits"]["nvidia.com/gpu"] = str(n_gpus_per_worker)
            if n_cpus_per_worker:
                ctr["resources"]["limits"]["cpu"] = str(n_cpus_per_worker)

    cluster = KubeCluster(
        custom_cluster_spec=custom_cluster_spec, shutdown_on_close=False
    )
    print(f"{cluster = }")


if __name__ == "__main__":

    def parse_pvcs(specs: str) -> dict[str, str]:
        name_to_path = {}
        for pvc in specs.split(","):
            # Can be empty
            if not pvc:
                continue
            name, _, path = pvc.partition(":")
            name_to_path[name] = path
        return name_to_path

    parser = argparse.ArgumentParser()
    ArgumentHelper(parser).add_args_create_k8s_dask_cluster()

    args = parser.parse_args()

    create_cluster(
        **vars(args),
    )
