import os
from modelWrapper import *
from Box_det import *
from Training_Case import *
from Utils import *

train_list = df2training_list(box_df)
train_X = np.concatenate([x.get_inputs() for x in train_list],axis=0)
train_y = np.concatenate([x.get_img() for x in train_list],axis=0)
#print('X shape {} Y shape {}: ', train_X.shape, train_y.shape)

if not os.path.exists('car_detection_team27'):
    model = build_model()
    model.fit(train_X,train_y,epochs=150)
    model.save('car_detection_team27')
else:
    model = tf.keras.models.load_model('car_detection_team27')

wrapper = modelWrapper(model)
    
testing_list = load_test_images(testing_dataset_path)

for case in testing_list:
    get_box(wrapper,case)

for i in range(5):
    idx = random.randrange(len(testing_list))
    while len(testing_list[idx].boxes) == 0:
        idx = random.randrange(len(testing_list))
    testing_list[idx].draw_image_with_boxes()