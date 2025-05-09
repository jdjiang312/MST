import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from uuid import uuid4
from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        fp_meter = AverageMeter() 
        fn_meter = AverageMeter() 
        tn_meter = AverageMeter()  

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data.cpu().numpy()
                np.save(pred_save_path, pred)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]

            # Updated to get TP, FP, FN, TN
            intersection, union, target, fp, fn, tn = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            fp_meter.update(fp) 
            fn_meter.update(fn) 
            tn_meter.update(tn) 

            record[data_name] = dict(
                intersection=intersection, union=union, target=target, fp=fp, fn=fn, tn=tn
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

            # Log TP, FP, FN, TN
            logger.info(
                f"Class-wise metrics for {data_name}:"
                f" TP={intersection.tolist()}, FP={fp.tolist()}, FN={fn.tolist()}, TN={tn.tolist()}"
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            fp = np.sum([meters["fp"] for _, meters in record.items()], axis=0)
            fn = np.sum([meters["fn"] for _, meters in record.items()], axis=0)
            tn = np.sum([meters["tn"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )

            # Log aggregated TP, FP, FN, TN
            logger.info(f"Aggregated metrics across all classes:")
            logger.info(f"TP={intersection.tolist()}, FP={fp.tolist()}, FN={fn.tolist()}, TN={tn.tolist()}")

            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch



def evaluate_completeness_commission(scenes, segment_ignore_index=(-1,), iou_thresh=0.5):

    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    for scene in scenes:
        pred_visited = dict()
        gt_part = scene["gt"]
        pred_part = scene["pred"]

        gt_list = []
        for cname, cinsts in gt_part.items():
            gt_list.extend(cinsts)

        pred_list = []
        for cname, cinsts in pred_part.items():
            pred_list.extend(cinsts)
            for p in cinsts:
                pred_visited[p["uuid"]] = False

        num_gt = len(gt_list)
        num_pred = len(pred_list)

        scene_true_positives = 0
        for gt_inst in gt_list:
            found_match = False
            for matched_pred in gt_inst.get("matched_pred", []):
                if pred_visited[matched_pred["uuid"]]:
                    continue

                overlap = float(matched_pred["intersection"]) / float(
                    gt_inst["vert_count"] + matched_pred["vert_count"] - matched_pred["intersection"] + 1e-10
                )
                if overlap >= iou_thresh:
                    scene_true_positives += 1
                    pred_visited[matched_pred["uuid"]] = True
                    found_match = True
                    break
            if not found_match:
                pass

        scene_false_negatives = num_gt - scene_true_positives
        used_pred = sum(pred_visited.values()) 
        scene_false_positives = num_pred - used_pred

        all_true_positives += scene_true_positives
        all_false_negatives += scene_false_negatives
        all_false_positives += scene_false_positives

    total_gt = all_true_positives + all_false_negatives
    total_pred = all_true_positives + all_false_positives
    recall = float(all_true_positives) / float(total_gt + 1e-10)
    precision = float(all_true_positives) / float(total_pred + 1e-10)
    completeness = recall
    omission_error = 1.0 - recall
    commission_error = 1.0 - precision
    f1 = 2.0 * recall * precision / (recall + precision + 1e-10)

    return dict(
        tp=all_true_positives,
        fp=all_false_positives,
        fn=all_false_negatives,
        recall=recall,
        precision=precision,
        completeness=completeness,
        omission=omission_error,
        commission=commission_error,
        f1_score=f1,
    )


@TESTERS.register_module()
class InsSegTester(TesterBase):
    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=False,
        segment_ignore_index=(-1,),
        instance_ignore_index=-1,
    ):
        super().__init__(cfg, model, test_loader, verbose)
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(self.save_path)

    def associate_instances(self, pred, segment, instance):

        segment = segment.cpu().numpy().astype(np.int32)
        instance = instance.cpu().numpy().astype(np.int32)
        void_mask = np.in1d(segment, self.segment_ignore_index)

        gt_instances = {}
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []

        instance_ids, idx, counts = np.unique(instance, return_index=True, return_counts=True)
        segment_ids = segment[idx] 
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict(
                instance_id=instance_ids[i],
                segment_id=segment_ids[i],
                vert_count=counts[i],
                matched_pred=[]
            )
            class_name = self.cfg.data.names[segment_ids[i]]
            gt_instances[class_name].append(gt_inst)

        pred_instances = {}
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []

        instance_id = 0
        num_pred = len(pred["pred_classes"])
        for i in range(num_pred):
            c_i = pred["pred_classes"][i]
            if c_i in self.segment_ignore_index:
                continue

            if isinstance(pred["pred_masks"][i], torch.Tensor):
                pmask_np = pred["pred_masks"][i].cpu().numpy().astype(bool)
            else:
                pmask_np = pred["pred_masks"][i].astype(bool)

            vert_count = np.count_nonzero(pmask_np)
            void_intersect = np.count_nonzero(np.logical_and(void_mask, pmask_np))

            pred_inst = dict(
                uuid=uuid4(),
                instance_id=instance_id,
                segment_id=c_i,
                confidence=pred["pred_scores"][i],
                mask=pmask_np,
                vert_count=vert_count,
                void_intersection=void_intersect,
                matched_gt=[]
            )
            if vert_count < 1:
                continue

            class_name = self.cfg.data.names[c_i]
            for gt_ in gt_instances[class_name]:
                intersection = np.count_nonzero(np.logical_and(instance == gt_["instance_id"], pmask_np))
                if intersection > 0:
                    gt_copy = dict(gt_)
                    gt_copy["intersection"] = intersection

                    pred_inst_copy = dict(pred_inst)
                    pred_inst_copy["intersection"] = intersection

                    gt_["matched_pred"].append(pred_inst_copy)
                    pred_inst["matched_gt"].append(gt_copy)
            pred_instances[class_name].append(pred_inst)
            instance_id += 1

        return gt_instances, pred_instances

    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Instance Test >>>>>>>>>>>>>>>>")
        self.model.eval()
        comm.synchronize()

        all_scenes = []
        batch_time = AverageMeter()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()

            data_dict = data_dict[0]
            data_name = data_dict["name"]

            fragment_list = data_dict["fragment_list"]
            frag0 = fragment_list[0] 
            segment = frag0["segment"]
            instance = frag0["instance"]

            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.cuda(non_blocking=True)
            for k, v in frag0.items():
                if isinstance(v, torch.Tensor):
                    frag0[k] = v.cuda(non_blocking=True)

            with torch.no_grad():
                pred = self.model(frag0)

            N = segment.size(0)
            device = segment.device
            pred_inst_class = torch.full((N,), -1, dtype=torch.int32, device=device)

            num_pred = len(pred["pred_masks"])  
            for i in range(num_pred):
                pmask_t = pred["pred_masks"][i].bool().to(device)
                pred_inst_class[pmask_t] = i + 1

            if "inverse" in frag0:
                inverse_idx = frag0["inverse"]  
                if inverse_idx.device != pred_inst_class.device:
                    inverse_idx = inverse_idx.to(pred_inst_class.device)

                pred_inst_class_cpu = pred_inst_class.cpu().numpy()
                inverse_idx_cpu = inverse_idx.cpu().numpy()
                # full_pred shape=(origN,)
                full_pred = pred_inst_class_cpu[inverse_idx_cpu]

                np_save_path = os.path.join(self.save_path, f"{data_name}_pred.npy")
                np.save(np_save_path, full_pred.astype(np.int32))
                logger.info(f"[UpSample] saved {full_pred.shape} => {np_save_path}")
            else:
                pred_inst_class_cpu = pred_inst_class.cpu().numpy().astype(np.int32)
                np_save_path = os.path.join(self.save_path, f"{data_name}_pred.npy")
                np.save(np_save_path, pred_inst_class_cpu)
                logger.info(f"[No UpSample] saved {pred_inst_class_cpu.shape} => {np_save_path}")

            gt_instances, pred_instances = self.associate_instances(pred, segment, instance)
            all_scenes.append(dict(gt=gt_instances, pred=pred_instances))

            batch_time.update(time.time() - end)
            logger.info(
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-Num {N}, "
                f"BatchTime {batch_time.val:.3f} (avg: {batch_time.avg:.3f})"
            )

        logger.info("Syncing across all GPUs ...")
        comm.synchronize()

        all_scenes_sync = comm.gather(all_scenes, dst=0)
        if comm.is_main_process():
            final_scenes = []
            for scenes_part in all_scenes_sync:
                final_scenes.extend(scenes_part)
            metrics = evaluate_completeness_commission(
                final_scenes, segment_ignore_index=self.segment_ignore_index, iou_thresh=0.5
            )
            logger.info("===== Final Metrics (IoU>=0.5) =====")
            logger.info(
                f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}\n"
                f"Completeness(Recall)={metrics['completeness']:.4f}, "
                f"Omission Error={metrics['omission']:.4f}, "
                f"Commission Error={metrics['commission']:.4f}, "
                f"F1={metrics['f1_score']:.4f}"
            )
            logger.info(">>>>>>>>>>>>>>>> End Instance Test  <<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


