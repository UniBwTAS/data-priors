from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv
import os
from filelock import FileLock, Timeout
from pathlib import Path


class DistributedTensorBoardSummaryWriter:
    def __init__(self, accelerator, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.accelerator = accelerator

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.accelerator.is_main_process:
            self.writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.accelerator.is_main_process:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def gather_add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        scalar_value = self.accelerator.gather(scalar_value).mean()
        self.add_scalar(tag, scalar_value, global_step, walltime)

    def gather_add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        tag_scalar_dict = {k: self.accelerator.gather(v).mean() for k, v in tag_scalar_dict.items()}
        self.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.writer.add_image(tag, img_tensor, global_step, walltime, dataformats)

    def close(self):
        self.writer.close()


class CSVLogger:
    def __init__(self, exper_name, accelerator, csv_path="experiments/summaries.csv"):

        self.accelerator = accelerator
        self.exper_name = exper_name
        self.csv_path = csv_path

        if accelerator.is_main_process:
            default_headers = [
                "name", "date", "pretrained", "pretrained0",  "dataset", "train_with_val", "fitness", "iou",
                "accuracy", "loss", "optimizer", "learning_rate", "weight_decay", "resume",
                "freeze_backbone", "master_labels", "batch_size", "warm_up",
                "init_head", "model", "epoch"
            ]
            # Create the CSV file with headers if it doesn't exist
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}. Creating new one.")
                with open(csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(default_headers)

            self.data = self._read_csv(csv_path)

    def _read_csv(self, csv_path):
        """Reads the CSV and returns the row where 'name' equals exper_name."""
        if self.accelerator.is_main_process and os.path.exists(csv_path):
            with open(csv_path, mode='r') as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames

                for row in reader:
                    if row["name"] == self.exper_name:
                        return row

                return {header: "" for header in headers}

    def update_config(self, config):
        """Updates the config data in the CSV."""
        if self.accelerator.is_main_process:
            pretrained = config.get("pretrained", "")
            pretrained0 = config.get("pretrained0", "")
            dataset = str(config["dataset"])
            train_with_val = str(config.get("train_with_val", ""))
            optim = config["optimizer"]
            optim_name = str(optim["name"])
            learning_rate = str(optim[optim_name]["lr"])
            weight_decay = str(optim[optim_name].get("weight_decay", ""))
            resume = str(config.get("resume", ""))
            freeze_backbone = str(optim.get("freeze_backbone", ""))
            master_labels = config["master_labels"]
            batch_size = str(config["training_params"]["batch_size"])
            warm_up = str(config["lr_scheduler"].get("warmup_epochs", 0))
            init_head = str(config.get("init_head", ""))
            model = str(config["model"])
            self.data.update({
                "name": self.exper_name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pretrained": pretrained,
                "pretrained0": pretrained0,
                "dataset": dataset,
                "train_with_val": train_with_val,
                "optimizer": optim_name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "resume": resume,
                "freeze_backbone": freeze_backbone,
                "master_labels": master_labels,
                "batch_size": batch_size,
                "warm_up": warm_up,
                "init_head": init_head,
                "model": model
            })
            self.write_to_file()

    def update_metrics(self, metrics):
        """Updates the metrics data in the CSV."""
        if self.accelerator.is_main_process:
            self.data.update(metrics)
            self.write_to_file()

    def write_to_file(self):
        """Writes the current data to the CSV file, updating the row where 'name' matches exper_name."""

        if self.accelerator.is_main_process:

            # lock file to avoid race conditions when multiple trainings are running simultaneously
            csv_path = Path(self.csv_path)
            lock_path = csv_path.with_name(".~lock." + csv_path.name + "#")
            lock = FileLock(lock_path, timeout=10)

            try:
                with lock:
                    updated = False
                    all_rows = []

                    with open(self.csv_path, mode='r') as file:
                        reader = csv.DictReader(file)
                        headers = reader.fieldnames

                        for row in reader:
                            if row["name"] == self.exper_name:
                                all_rows.append(self.data)
                                updated = True
                            else:
                                all_rows.append(row)

                    if not updated:
                        all_rows.append(self.data)

                    with open(self.csv_path, mode='w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(all_rows)

            except Timeout:
                print(f"Could not acquire the file lock for {self.csv_path} within the timeout period.")
            except Exception as e:
                print(f"An error occurred while writing to the CSV file: {e}")