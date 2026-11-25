import os
import cv2
import numpy as np

def prepare_chest_xray_data(data_root, num_per_label=200, target_size=512, random_seed=42):
    np.random.seed(random_seed)
    img_list = []
    label_list = []
    label_dict = {
        "NORMAL": -1,
        "PNEUMONIA": 1
    }

    for class_name in label_dict:
        class_dir = os.path.join(data_root, class_name)
        file_names = sorted(os.listdir(class_dir))
        valid_imgs = []
        for fname in file_names:
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                valid_imgs.append(fname)

        selected_imgs = np.random.choice(valid_imgs, size=num_per_label, replace=False)
        target_val = label_dict[class_name]

        for img_name in selected_imgs:
            full_path = os.path.join(class_dir, img_name)
            gray_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue
            resized_img = cv2.resize(gray_img, (target_size, target_size))
            flat_vec = resized_img.flatten() / 255.0
            img_list.append(flat_vec)
            label_list.append(target_val)

    X_data = np.array(img_list, dtype=np.float64)
    a_data = np.array(label_list, dtype=np.int64)

    shuffle_idx = np.random.permutation(len(X_data))
    X_data = X_data[shuffle_idx]
    a_data = a_data[shuffle_idx]
    return X_data, a_data

if __name__ == "__main__":
    X_check, a_check = prepare_chest_xray_data("data/chest_xray/train", num_per_label=10, target_size=512)
    print("X shape:", X_check.shape)
    print("a shape:", a_check.shape)
    print("Labels:", a_check)