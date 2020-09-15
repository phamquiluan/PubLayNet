import os
import re
import sys
import json
import glob
import bisect
import logging
import platform
from subprocess import Popen, PIPE
from multiprocessing import Process

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from natsort import natsorted
from tqdm import tqdm


logger = logging.getLogger(__name__)

def _get_command_path(command):
    if platform.system() == 'Windows':
        command = command + '.exe'
    return command


def _pdf_info(pdf_path):
    command = [_get_command_path("pdfinfo"), pdf_path]
    proc = Popen(command, stdout=PIPE, stderr=PIPE)
    out = proc.communicate()[0]

    page_count = int(re.search(r'Pages:\s+(\d+)', out.decode("utf8", "ignore")).group(1))
    pdf_info = []
    for i in range(page_count):
        cmd = [_get_command_path("pdfinfo"),
               '-f', str(i + 1), '-l', str(i + 1), pdf_path]
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
        out = proc.communicate()[0]
        # Page size: 792 x 1224 pts
        page_size = re.search('Page.*size:\s+([\d\.x\s]+)',
                              out.decode("utf8", "ignore")).group(1).split('x')
        pdf_info.append((str(i + 1), list(map(float, page_size))))

    return pdf_info


def pdf_to_image(
        pdf_path="./tmp/hihi/GED.pdf",
        threadnum=8,
        dpi=300
    ):
    assert 'pdf' in pdf_path
    page_nums = [pnum for (pnum, _) in _pdf_info(pdf_path)]

    processes = []

    image_dir = pdf_path[:-4]
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    print("Generating images for", pdf_path)
    for page_num in tqdm(
            page_nums,
            total=len(page_nums),
            leave=False
        ):
        command = [_get_command_path("pdftoppm"),
                   "-cropbox", "-png", "-singlefile",
                   "-r", str(dpi), "-f", page_num,
                   "-l", page_num, pdf_path,
                   os.path.join(image_dir, page_num)]
        processes.append(Popen(command, stdout=PIPE, stderr=PIPE))

        if len(processes) == threadnum:
            processes[0].wait()
            processes.pop(0)

    for proc in processes:
        proc.wait()


def _get_pdf_page_count(pdf_path):
    command = [_get_command_path("pdfinfo"), pdf_path]
    proc = Popen(command, stdout=PIPE, stderr=PIPE)
    out = proc.communicate()[0]

    page_count = int(re.search(r"Pages:\s+(\d+)", out.decode("utf8", "ignore")).group(1))
    return page_count


def get_textline(pdf_path, dpi=300):
    assert "pdf" in pdf_path and os.path.exists(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    pdf_page_num = _get_pdf_page_count(pdf_path)

    output = {}
    
    print("Generating textline for", pdf_path)
    
    for page_num in range(1, pdf_page_num + 1):
        command = [
            "pdftotext",
            "-r", str(dpi),
            "-f", str(page_num),
            "-l", str(page_num),
            "-bbox-layout",
            pdf_path, "-"
        ]

        p = Popen(command, stdout=PIPE)
        stdout, _ = p.communicate()
        stdout = stdout.decode("utf-8")
        
        # <line xMin="3835.000000" yMin="2381.166667" xMax="5008.999500" yMax="2418.704167">
        line_pattern = re.compile(r"<line xMin=\"([\d.]+)\" yMin=\"([\d.]+)\" xMax=\"([\d.]+)\" yMax=\"([\d.]+)\">")
        line_boxes = line_pattern.findall(stdout)
        
        textline_list = []
        for xmin, ymin, xmax, ymax in line_boxes:
            xmin, ymin, xmax, ymax = list(map(float, [xmin, ymin, xmax, ymax ]))
            xmin, ymin, xmax, ymax = list(map(int, [xmin, ymin, xmax, ymax ]))

            
            """
            top = xmin - 2 
            left = ymin - 2 
            width = xmax - top  + 4
            height = ymax - left + 4 

            command = [
                "pdftotext",
                "-r", str(dpi),
                "-f", str(page_num),
                "-l", str(page_num),
                "-x", str(top),
                "-y", str(left),
                "-W", str(width),
                "-H", str(height),
                pdf_path, "-"
            ]

            p = Popen(command, stdout=PIPE)
            stdout, _ = p.communicate()

            ocr_value = stdout.decode("utf-8")
            ocr_value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', ocr_value).strip()
            
            output.append({
                "type": "textline",
                "location": [xmin, ymin, xmax, ymax],
                "content": ocr_value,
                "page": page_num
            })
            """

            textline_list.append({
                "type": "textline",
                "location": [xmin, ymin, xmax, ymax]
            })
        
        if len(textline_list) > 0:
            output[str(page_num)] = textline_list

    return output



class TeeNet():
    def __init__(self, root_dir="/data/teenet/", transforms=None):
        print("=> Initializing TeeNet..")
        self.root_dir = root_dir
        self.pdf_dir = os.path.join(root_dir, "pdf")
        self.total_pdf_files = len(glob.glob(os.path.join(self.pdf_dir, "*.pdf")))

        print("=> Found {} pdf files..".format(self.total_pdf_files))

        self.transforms = transforms

        assert os.path.exists(root_dir), "{} does not exists!".format(root_dir)
        assert os.path.exists(self.pdf_dir), "{} does not exists!".format(self.pdf_dir)

        # == spawn processes to generate textlines
        """
        processes = []
        for idx, pdf_path in enumerate(glob.glob(os.path.join(self.pdf_dir, "*.pdf"))):
            pdf_name = os.path.basename(pdf_path)
            print("Generate textline for {}".format(pdf_name))

            p = Process(target=self.gen_labels, args=(pdf_path,))
            p.start()
            processes.append(p)
        [p.join() for p in processes]
        """
        # ======
        
        # indexing, maker hierarchy for anns
        self.indexing()


    def indexing(self):
        """ 
        create anns structure based on textline json
        """
        print("==> Start indexing TeeNet..")
        self.anns = []
        self.page_counts = []
        
        pre = 0
        # for label_path in natsorted(glob.glob(os.path.join(self.root_dir, "json", "*.json"))):
        for label_path in natsorted(glob.glob(os.path.join(self.root_dir, "new_json1", "*.json"))):

            label_name = os.path.basename(label_path)
            pdf_name = os.path.splitext(label_name)[0]

            """
            # checking if all image is good
            is_bad = False        
            lenght_pdf = len(glob.glob(os.path.join(self.root_dir, "pdf", pdf_name[:-4], "*.png")))
            for image_path_idx, image_path in enumerate(glob.glob(os.path.join(self.root_dir, "pdf", pdf_name[:-4], "*.png"))):

                try:
                    image = cv2.imread(image_path)
                    if image is None: 
                        is_bad = True

                    tmp_h, tmp_w = image.shape[:2]
                except Exception as e:
                    # ignore file if it touch any exception 
                    print(e)
                    is_bad = True
            
                if is_bad is True:
                    print("{}/{} BAD {}".format(image_path_idx, lenght_pdf, os.path.basename(image_path)))
                    break

                print("{}/{} GOOD {}".format(image_path_idx, lenght_pdf, os.path.basename(image_path)))

            if is_bad is True:
                is_bad = False
                continue
            """

            with open(label_path, "r") as label_file_ref:
                label = json.load(label_file_ref)

                if len(label) == 0:  # there is nothing in this file
                    pass

                else:  # create list of anns for this file
                    """
                    self.anns = [
                        {
                            "pdf_name": "sample.pdf",
                            "page_idx_list": [1, 2],
                            "page_textline_list": [[..], [..], [..]]
                        }, ..
                    ]
                    """

                    ann = {
                        "pdf_name": pdf_name,
                        "page_idx_list": [],
                        "page_size_list": [],
                        "page_textline_list": []
                    }

                    # for page_idx, page_textline_list in label.items():
                    # for page_data in label.items():
                    for page_data in label:
                        # ann["page_idx_list"].append(page_idx)
                        # ann["page_textline_list"].append(page_textline_list) 

                        ann["page_idx_list"].append(page_data["page_num"])
                        ann["page_size_list"].append(page_data["page_size"])
                        ann["page_textline_list"].append(page_data["page_text_list"]) 

                    self.anns.append(ann)
                    
                    page_count = len(ann["page_idx_list"])
                    self.page_counts.append(pre + page_count)
                    pre += page_count

        if len(self.page_counts) > 0:
            logger.info("we have total {} pages".format(self.page_counts[-1]))
        else:
            print("we have no pages, exit..")
            raise SystemExit

    def gen_labels(self, pdf_path):
        # NOTE: currently, this function is redundant and it not maintained correctly
        pdf_name = os.path.basename(pdf_path)
        
        if not os.path.exists(os.path.join(self.root_dir, "json")):
            os.makedirs(os.path.join(self.root_dir, "json"), exist_ok=True)

        if not os.path.exists(os.path.join(self.root_dir, "json")):
            os.makedirs(os.path.join(self.root_dir, "json"), exist_ok=True)

        label_path = os.path.join(self.root_dir, "json", "{}.json".format(pdf_name))

        if not os.path.exists(label_path):
            labels = get_textline(pdf_path)
            with open(label_path, "w") as label_file_ref:
                json.dump(labels, label_file_ref)
                del labels
    
    def __len__(self):
        return self.page_counts[-1]
    
    def get_height_and_width(self, idx):
        _, _, page_size, _ = self.get_ann(idx)
        # image_path = os.path.join(self.root_dir, "pdf", pdf_name[:-4], "{}.png".format(page_num))
        # pdf_path = os.path.join(self.root_dir, "pdf", pdf_name)

        # NOTE does not need to gen image anymore
        # if not os.path.exists(image_path): # sinh anh thoi ba con oi
        #     pdf_to_image(pdf_path, threadnum=os.cpu_count())

        # well, we have anns here

        # image = cv2.imread(image_path)
        # if image is None: 
        #     # hard remove this pdf file
        #     pdf_to_image(pdf_path, threadnum=os.cpu_count())
        #     image = cv2.imread(image_path)
        return page_size[0], page_size[1]
        # return image.shape[0], image.shape[1]

    def get_ann(self, idx):
        """
        follow this formular
        idx = self.page_counts[acc_prev_pdf] + page_num_idx
        """
        cur = bisect.bisect_right(self.page_counts, idx)
        prev = cur - 1 if cur > 0 else None
        
        page_idx = idx
        if prev is not None:
            page_idx -= self.page_counts[prev]

        pdf_name = self.anns[cur]["pdf_name"]
        page_num = self.anns[cur]["page_idx_list"][int(page_idx)]
        page_size = self.anns[cur]["page_size_list"][int(page_idx)]
        text_list = self.anns[cur]["page_textline_list"][int(page_idx)]

        return pdf_name, page_num, page_size, text_list

    def __getitem__(self, idx):
        
        pdf_name, page_num, page_size, text_list = self.get_ann(idx)

        # print(pdf_name, page_num)
        # image_name, labels_boxes = self.anns[0]

        image_path = os.path.join(self.root_dir, "pdf", pdf_name[:-4], "{}.png".format(page_num))
        pdf_path = os.path.join(self.root_dir, "pdf", pdf_name)

        if not os.path.exists(image_path): # sinh anh thoi ba con oi
            pdf_to_image(pdf_path, threadnum=os.cpu_count())

        image = cv2.imread(image_path)
        # logger.info("original image.shape {}".format(image.shape))

        scale_factor = 1200 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        # logger.info("new image.shape {}".format(image.shape))
        # logger.info("scale factor {}".format(scale_factor))


        boxes = []
        labels = []
        instance_masks = []
        iscrowd = []

        # for box in labels_boxes:

        # logger.info("{}:{} -> text num:{}".format(pdf_name, page_num, len(text_list)))
        for text_idx, box in enumerate(text_list):
            if text_idx > 400:  # just accept 400 textlint in a document
                break

            xmin, ymin, xmax, ymax = box["location"]

            # logger.info("original box {}".format(box["location"]))

            xmin *= scale_factor 
            ymin *= scale_factor 
            xmax *= scale_factor 
            ymax *= scale_factor 
            xmin, ymin, xmax, ymax = list(map(int, [xmin, ymin, xmax, ymax]))


            # logger.info("new box {}".format([xmin, ymin, xmax, ymax]))

            # assert xmin < xmax and ymin < ymax, "{} {} {} {}".format(xmin, ymin, xmax, ymax)
            # assert 0 <= xmin <= image.shape[1], "image shape {}, {}".format(image.shape, xmin)
            # assert 0 <= xmax <= image.shape[1], "image shape {}, {}".format(image.shape, xmax)
            # assert 0 <= ymin <= image.shape[0], "image shape {}, {}".format(image.shape, ymin)
            # assert 0 <= ymax <= image.shape[0], "image shape {}, {}".format(image.shape, ymax)

            xmin = max(0, min(xmin, image.shape[1]))
            xmax = max(0, min(xmax, image.shape[1]))
            ymin = max(0, min(ymin, image.shape[0]))
            ymax = max(0, min(ymax, image.shape[0]))

            if xmin >= xmax or ymin >= ymax:
                # logger.warn("Ignore box {}".format([xmin, ymin, xmax, ymax]))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
            instance_mask = np.zeros((image.shape[0], image.shape[1]))
            instance_mask[ymin:ymax, xmin:xmax] = 1
            instance_masks.append(instance_mask)

            assert instance_mask.shape == image.shape[:2]
            iscrowd.append(0)


        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # labels = [a["category_id"] for a in anns]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # instance_masks = [self.coco.annToMask(ann) for ann in anns]
        instance_masks = np.array(instance_masks)
        instance_masks = torch.as_tensor(instance_masks, dtype=torch.uint8)

        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # iscrowd = [a["iscrowd"] for a in anns]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = instance_masks
        # target["image_id"] = torch.tensor([anns[0]["image_id"]])
        # target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


if __name__ == "__main__":
    print("Starting main..")
    dataset = TeeNet(root_dir="/data/tiny_teenet")

    for i in range(len(dataset)):
        image, target = dataset[i]
        print(image.shape)
