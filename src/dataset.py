import os
import open3d as o3d
import numpy as np
import torch


class Ds_point_model:

    def __init__(self, root="Shapenet"):
        self.root = root
        self.classes = []
        self.dict = {}
        self.all_files = []

        if not os.path.exists(root):
            print(f"Warning: Root directory '{root}' does not exist.")
            return

        for rt, dirs, files in os.walk(root):
            self.classes = dirs
            for cl in dirs:
                self.dict[cl] = []
            break

        for cl in self.classes:
            class_path = os.path.join(self.root, cl)
            for sub_rt, _, sub_files in os.walk(class_path):
                for file in sub_files:
                    if file.endswith(".obj"):
                        full_path = os.path.join(sub_rt, file)
                        self.dict[cl].append(full_path)

                        self.all_files.append((cl, full_path))

    @staticmethod
    def map():
        mapp = {
            "04379243": "table",
            "03593526": "jar",
            "04225987": "skateboard",
            "02958343": "car",
            "02876657": "bottle",
            "04460130": "tower",
            "03001627": "chair",
            "02871439": "bookshelf",
            "02942699": "camera",
            "02691156": "airplane",
            "03642806": "laptop",
            "02801938": "basket",
            "04256520": "sofa",
            "03624134": "knife",
            "02946921": "can",
            "04090263": "rifle",
            "04468005": "train",
            "03938244": "pillow",
            "03636649": "lamp",
            "02747177": "trash bin ",
            "03710193": "mailbox",
            "04530566": "watercraft",
            "03790512": "motorbike",
            "03207941": "dishwasher",
            "02828884": "bench",
            "03948459": "pistol",
            "04099429": "rocket",
            "03691459": "loudspeaker",
            "03337140": "file cabinet",
            "02773838": "bag ",
            "02933112": "cabinet",
            "02818832": "bed",
            "02843684": "birdhouse",
            "03211117": "display",
            "03928116": "piano",
            "03261776": "earphone",
            "04401088": "telephone",
            "04330267": "stove",
            "03759954": "microphone",
            "02924116": "bus",
            "03797390": "mug",
            "04074963": "remote",
            "02808440": "bathtub",
            "02880940": "bowl",
            "03085013": "keyboard",
            "03467517": "guitar",
            "04554684": "washer",
            "02834778": "bicycle",
            "03325088": "faucet",
            "04004475": "printer",
            "02954340": "cap",
            "02992529": "celular",
        }

        return mapp

    def __len__(self):

        return len(self.all_files)

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.all_files):
            raise IndexError("Dataset index out of range.")

        class_name, file_path = self.all_files[idx]

        return class_name, file_path


class Ds_point_sampled:
    def __init__(self, model: Ds_point_model):
        self.model = model
        if not os.path.exists("point_clouds"):
            os.makedirs("point_clouds")
            self.save_all()

    def save_all(self):
        for i in range(len(self.model)):
            class_name, file_path = self.model[i]
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
            save_path = "point_clouds/" + f"{i}.ply"
            o3d.io.write_point_cloud(save_path, pcd)

    def __getitem__(self, idx):
        class_name, file_path = self.model[idx]
        file_path = "point_clouds/" + f"{idx}.ply"
        pcd = o3d.io.read_point_cloud(file_path)

        points = np.asarray(pcd.points, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        features = np.concatenate([points, normals], axis=1)

        return class_name, torch.from_numpy(features)

    def __len__(self):
        return len(self.model)
