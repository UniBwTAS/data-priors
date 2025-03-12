import torch


class SegmentationEvaluator:
    def __init__(self, num_classes, device, label_map=None, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64).to(device)
        self.label_map = label_map

        # Metrics
        self.IoU = None
        self.mIoU = None
        self.mPA = None
        self.mCA = None

    def add_batch(self, predictions, ground_truth, pixel_mask):
        if len(predictions.shape) == 4:  # Convert from probabilities to class indices
            predictions = torch.argmax(predictions, dim=1)

        predictions = predictions.flatten().long()
        ground_truth = ground_truth.flatten().long()
        pixel_mask = pixel_mask.flatten().bool()

        valid = (ground_truth != self.ignore_index) & pixel_mask
        predictions = predictions[valid]
        ground_truth = ground_truth[valid]

        with torch.no_grad():
            k = (ground_truth >= 0) & (ground_truth < self.num_classes)
            inds = self.num_classes * ground_truth[k] + predictions[k]
            bins = torch.bincount(inds, minlength=self.num_classes ** 2)
            bins = bins[:self.num_classes ** 2]  # Ensure bins tensor is not longer than needed
            self.confusion_matrix += bins.reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        intersection = torch.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - intersection
        IoU = intersection / (total + 1e-10)  # Avoid division by zero
        mIoU = IoU.mean()

        self.IoU, self.mIoU = IoU, mIoU

    def compute_accuracy(self):
        # Assuming compute_iou has already been called
        total_correct_pixels = torch.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()

        # Pixel Accuracy
        mPA = total_correct_pixels.float() / total_pixels.float()

        # Mean Class Accuracy
        # Avoid division by zero by adding a small epsilon where the sum is zero
        sum_per_class = self.confusion_matrix.sum(1).float()
        epsilon = 1e-10  # Small constant
        valid = sum_per_class > 0  # Check which classes have at least one instance in the ground truth
        per_class_accuracy = torch.where(valid, torch.diag(self.confusion_matrix).float() / (sum_per_class + epsilon),
                                         torch.zeros_like(sum_per_class))
        # mCA = per_class_accuracy[valid].mean()  # Only average over valid classes

        self.mPA, self.mCA = mPA, per_class_accuracy

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64).to(self.device)
