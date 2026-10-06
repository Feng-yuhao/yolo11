"""Fixed, same-seed references; not tunable training hyperparameters."""
SEED = 2
BASE_REPORT = 'e1_yolo11s_p2add_img640_seed2_20260926_172907_911324'
PARENT_REPORT = 'E40S2_RUN_20260930_130808_314014'
BASE_RUN = 'e1_yolo11s_p2add_img640_seed2'
PARENT_RUN = 'e40a_spatial_task_head_img640_seed2_200e'
EXPERIMENT = 'e47a_p2_depth_readout_img640_seed2_200e'
BASE_EXPECTED = dict(AP50_95_percent=25.41065733278743,
                     AP50_percent=44.312152547264084,
                     AP75_percent=24.698933563022347,
                     AR500_percent=37.60478146181616)
PARENT_EXPECTED = dict(AP50_95_percent=26.063860003172476,
                       AP50_percent=45.21397062058152,
                       AP75_percent=25.664076075372027,
                       AR500_percent=37.89717545479253)


def require_scores(actual, expected, label):
    for key, value in expected.items():
        if key not in actual or abs(actual[key] - value) > 1e-7:
            raise RuntimeError(label + ' official reference did not reproduce ' + key)
