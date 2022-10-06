import pandas as pd
import numpy as np

def map_event():
    # ['click_about', 'click_courseware', 'click_forum', 'click_info',
#       'click_progress', 'close_courseware', 'create_comment',
#       'create_thread', 'delete_comment', 'delete_thread', 'load_video',
#       'pause_video', 'play_video', 'problem_check',
#       'problem_check_correct', 'problem_check_incorrect', 'problem_get',
#       'problem_save', 'reset_problem', 'seek_video', 'stop_video',
#       'close_forum']
    df=pd.read_csv("../kdd2015_Mooc/preprocess/all_count_per_type.csv")
    dic = {
            'click_about':'general_info',
            'click_courseware':'courseware',
            'click_forum':'forum',
            'click_info':'general_info',
            'click_progress':'general_info',
            'close_courseware':'courseware',
            'create_comment':'forum',
            'create_thread':'forum',
            'delete_comment':'forum',
            'delete_thread':'forum',
            'load_video':'video',
            'pause_video':'video',
            'play_video':'video',
            'problem_check':'problem',
            'problem_check_correct':'problem',
            'problem_check_incorrect':'problem',
            'problem_get':'problem',
            'problem_save':'problem',
            'reset_problem':'problem',
            'seek_video':'video',
            'stop_video':'video',
            'close_forum':'forum'
            }

    df["event_class"] = df["event"].apply(lambda x:dic[x])
    df.to_csv("../kdd2015_Mooc/preprocess/all_count_per_type_transformed.csv", index=False)

    return df
if __name__ == "__main__":
    map_event()
