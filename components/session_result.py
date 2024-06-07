import pandas as pd
from datetime import datetime
import itertools

def result_to_df(counters, tabs, counter_column = False):
    # result_df = cache
    # result_df = [item for sublist in result_df for item in sublist]
    # fields = ['frame_num', 'track_id', 'class_name', 'tlwh', 'confidence']
    # result_df = pd.DataFrame(
    #     [{fn: getattr( f, fn ) for fn in fields} for f in result_df] )
    # if 'tlwh' in result_df.columns:
    #     result_df['top'], result_df['left'], result_df['width'], result_df['height'] = zip(
    #         *result_df.pop( 'tlwh' ) )
    # if counter_column:
    #     counter_id = [[i] * len( r ) for i, r in enumerate( cache )]
    #     counter_id = [item for sublist in counter_id for item in sublist]
    #     result_df.insert( 0, 'counter', counter_id )
    # return result_df
    tracked_objects = list(counters[0].track_history.keys())
    tracked_paths = list(counters[0].track_history.values())
    first_appear = [min(x) for x in counters[0].track_in_frame.values()]
    last_appear = [max(x) for x in counters[0].track_in_frame.values()]
    result_df = pd.DataFrame({'object_id':tracked_objects, 'first_frame':first_appear, 'last_frame':last_appear, 'track':tracked_paths, })
    for i,counter in enumerate(counters[::-1]):
            counted_list = [check_object_class(counter,o) for o in tracked_objects]
            result_df.insert( 1, f'counter_{i}', counted_list )
    time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_df.to_excel(f'results\\{tabs}.xlsx')
    return result_df

def check_object_class (counter, object):
    for c,o in counter.class_wise_count.items():
        if object in list(itertools.chain.from_iterable((counter.class_wise_count[c].values()))):
            return c
    return None