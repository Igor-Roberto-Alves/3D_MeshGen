import os
import open3d as o3d
import numpy as np
import torch
import tqdm
from torchvision.transforms import transforms



class JointAugment(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

    def __call__(self, features):

        xyz = features[:, :3]
        nrm = features[:, 3:]


        noise = torch.randn_like(xyz) * self.jitter_sigma
        noise = noise.clamp(-self.jitter_clip, self.jitter_clip)
        xyz_jittered = xyz + noise

        theta = torch.rand(1, device=features.device) * 2 * torch.pi
        c = torch.cos(theta)
        s = torch.sin(theta)

        rot = torch.tensor([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ], dtype=features.dtype, device=features.device)

        xyz_rotated = xyz_jittered @ rot.T
        nrm_rotated = nrm @ rot.T

        return torch.cat([xyz_rotated, nrm_rotated], dim=1)

data_augs = JointAugment()


class Ds_point_model:

    def __init__(self, root="Shapenet", augment = data_augs):
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

        for cl in tqdm.tqdm(self.classes, "Generating dataset file list"):
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
        for i in tqdm.tqdm(range(len(self.model))):
            class_name, file_path = self.model[i]
            mesh = o3d.io.read_triangle_mesh(file_path)
            print(file_path)
            print(mesh)
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
            save_path = "point_clouds/" + f"{class_name}_{i}.ply"
            o3d.io.write_point_cloud(save_path, pcd)

    def __getitem__(self, idx):
        class_name, file_path = self.model[idx]
        file_path = "point_clouds/" + f"{class_name}_{idx}.ply"
        pcd = o3d.io.read_point_cloud(file_path)

        points = np.asarray(pcd.points, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        features = np.concatenate([points, normals], axis=1)

        return class_name, torch.from_numpy(features)

    def __len__(self):
        count = 0
        for root, dirs, files in os.walk("point_clouds"):
            for filez in files:
                count += 1

        return count


class Ds_point_sampled_already:
    def __init__(self, root="point_clouds", augment=True):
        self.root = root
        all_files = []
        for rt, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".ply"):
                    all_files.append(os.path.join(rt, file))

        self.files = [f for f in all_files if os.path.exists(f) and os.path.getsize(f) > 0]
        skipped = len(all_files) - len(self.files)
        if skipped:
            print(f"[dataset] Skipped {skipped} missing/empty PLY files.")

        self.augment = augment
        self.transform = data_augs

        present_classes = sorted({
            os.path.basename(f).replace(".ply", "").split("_")[0] for f in self.files
        })
        self.class_to_idx = {cls_id: idx for idx, cls_id in enumerate(present_classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.basename(self.files[idx])
        class_name, _ = filename.replace(".ply", "").split("_")

        class_idx = self.class_to_idx.get(class_name, 0)
        cls_tensor = torch.tensor(class_idx, dtype=torch.long)
    
        file_path = self.files[idx]
        pcd = o3d.io.read_point_cloud(file_path)

        points = np.asarray(pcd.points, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)

        # Fallback: corrupt/unreadable file returns 0 points — pick a random valid sample
        if points.shape[0] == 0:
            return self.__getitem__(int(torch.randint(len(self), (1,)).item()))

        features = np.concatenate([points, normals], axis=1)
        features = torch.from_numpy(features).float()

        if self.augment and self.transform:
            features = self.transform(features)

        return features, cls_tensor


def generate_subset(shapenet_root: str, out_dir: str, classes: list[str], n_points: int = 2048):


    os.makedirs(out_dir, exist_ok=True)

    model = Ds_point_model(root=shapenet_root)
    subset = [(cls, path) for cls, path in model.all_files if cls in classes]

    print(f"Found {len(subset)} meshes across classes {classes}")
    skipped = 0
    saved = 0
    for i, (cls, file_path) in enumerate(tqdm.tqdm(subset, desc="Sampling")):
        save_path = os.path.join(out_dir, f"{cls}_{i}.ply")
        if os.path.exists(save_path):
            saved += 1
            continue
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            skipped += 1
            continue
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        o3d.io.write_point_cloud(save_path, pcd)
        saved += 1

    print(f"Done. {saved} saved, {skipped} skipped (unreadable meshes).")


if __name__ == "__main__":
    import argparse

    CLASSES = ["02691156", "02958343"]

    p = argparse.ArgumentParser(description="Generate PLY point clouds from ShapeNet OBJ meshes.")
    p.add_argument("--shapenet_root", type=str, required=True,
                   help="Path to ShapeNet root directory (contains class-id subdirs)")
    p.add_argument("--out_dir", type=str, default="point_clouds",
                   help="Output directory for PLY files (default: point_clouds)")
    p.add_argument("--n_points", type=int, default=2048,
                   help="Number of points to sample per mesh (default: 2048)")
    args = p.parse_args()

    generate_subset(
        shapenet_root=args.shapenet_root,
        out_dir=args.out_dir,
        classes=CLASSES,
        n_points=args.n_points,
    )