import os
import glob
import json
import cv2
import numpy as np

# function that reads json file, produces the gt image masks and places the right folder
def put_images_in_folder(json_label_path, src_training_dir, gt_image_dir, binary_seg_dir, instance_dir):
    # Read json files
    image_nums = len(os.listdir(gt_image_dir))
    print('Started Process {:s} success'.format(json_label_path))
    with open(json_label_path, 'r') as file:
        for index, line in enumerate(file):
            gt_info = json.loads(line)
            
            gt_lanes = gt_info['lanes']
            y_samples = gt_info['h_samples']
            raw_file = gt_info['raw_file']

            path_to_tusimple_image = os.path.join(src_training_dir, raw_file)

            new_image_name = '\{:s}.png'.format('{:d}'.format(index + image_nums).zfill(4))
            
            image_output_path = os.path.join(gt_image_dir + new_image_name)
            binary_output_path = os.path.join(binary_seg_dir + new_image_name)
            instance_output_path = os.path.join(instance_dir + new_image_name)
            
            src_image = cv2.imread(path_to_tusimple_image, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

            for lane in range(len(gt_lanes)):
                coordinate=[]
                for w,h in zip(gt_lanes[lane],y_samples):
                    if w==-2:
                        continue
                    else:
                        coordinate.append(np.array([w,h],dtype=np.int32))
            
                shadding=255-lane*50
                dst_instance_image=cv2.polylines(dst_instance_image,np.stack([coordinate]),isClosed=False,color=shadding,thickness=5)
                dst_binary_image=cv2.polylines(dst_binary_image,np.stack([coordinate]),isClosed=False,color=255,thickness=5)

            cv2.imwrite(binary_output_path, dst_binary_image)
            cv2.imwrite(instance_output_path, dst_instance_image)
            cv2.imwrite(image_output_path, src_image)  
        print('Finished Process {:s} success'.format(json_label_path))

# generate text file that stores paths to each gt image and masks
# ex C:\Users\anshul\Documents\school\ECE228\final_project\dataset\ground_truth_image\3625.png C:\Users\anshul\Documents\school\ECE228\final_project\dataset\ground_truth_binary_seg\3625.png C:\Users\anshul\Documents\school\ECE228\final_project\dataset\ground_truth_instance_seg\3625.png
def generate_text_file(ground_truth_image_dir, ground_truth_binary_seg_dir,ground_truth_instance_seg_dir, src_dir):
    with open(os.path.join(src_dir, "dataset_image_paths.txt"), 'w') as file:
        for image_name in os.listdir(ground_truth_image_dir):
            if not image_name.endswith('.png'):
                continue
            binary_gt_image_path = os.path.join(ground_truth_binary_seg_dir, image_name)
            instance_gt_image_path = os.path.join(ground_truth_instance_seg_dir, image_name)
            image_path = os.path.join(ground_truth_image_dir, image_name)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('Image set: {:s} broken'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info + '\n')

# parse the generated text file and split into validation and training splits
def generate_train_valid_split_txt():
    main_train_file_path = r'C:\Users\anshul\Documents\school\ECE228\final_project\dataset\dataset_image_paths.txt'

    train_split_file_path = r'C:\Users\anshul\Documents\school\ECE228\final_project\dataset\train_split.txt'
    valid_split_file_path = r'C:\Users\anshul\Documents\school\ECE228\final_project\dataset\valid_split.txt'

    with open(main_train_file_path, 'r') as file:
        data = file.readlines()
        train_data = data[0:int(len(data)*0.8)]
        valid_data = data[int(len(data)*0.8): -1]

    with open(train_split_file_path, 'w') as file:
        for d in train_data:
            file.write(d)
    with open(valid_split_file_path, 'w') as file:
        for d in valid_data:
            file.write(d)

# function to process the dataset
def process_dataset(src_training_dir, dst_training_dir):

    # Create folder structure
    os.makedirs(dst_training_dir, exist_ok=True)

    ground_truth_image_dir = os.path.join(dst_training_dir, 'ground_truth_image')
    ground_truth_binary_seg_dir = os.path.join(dst_training_dir, 'ground_truth_binary_seg')
    ground_truth_instance_seg_dir = os.path.join(dst_training_dir, 'ground_truth_instance_seg')

    os.makedirs(ground_truth_image_dir, exist_ok=True)
    os.makedirs(ground_truth_binary_seg_dir, exist_ok=True)
    os.makedirs(ground_truth_instance_seg_dir, exist_ok=True)

    # iterate thorugh the .json file in the training dataset directory and put images in correct folders
    for json_label_path in glob.glob('{:s}/label_data_*.json'.format(src_training_dir)):
        put_images_in_folder(json_label_path, src_training_dir, ground_truth_image_dir,ground_truth_binary_seg_dir,ground_truth_instance_seg_dir)

    # generate the text file that stores path gt images and masks
    generate_text_file(ground_truth_image_dir,ground_truth_binary_seg_dir, ground_truth_instance_seg_dir,dst_training_dir)
    # split into valid and test split
    generate_train_valid_split_txt()


process_dataset(r'C:\Users\anshul\Documents\school\ECE228\final_project\data\train_set',r'C:\Users\anshul\Documents\school\ECE228\final_project\dataset')


# For test
#process_dataset(r'C:\Users\anshul\Documents\school\ECE228\final_project\data\test_set',r'C:\Users\anshul\Documents\school\ECE228\final_project\dataset\test')
