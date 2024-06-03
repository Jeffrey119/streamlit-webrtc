import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import copy
from typing import List
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from supervision.annotators.utils import ColorLookup, Trace, resolve_color
from ultralytics import YOLO, solutions
from components import session_result
from collections import defaultdict

COLORS = np.random.uniform( 0, 255, size=(1024, 3) )

class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other: float):
        self.x *= other
        self.y *= other
        return self

    __rmul__ = __mul__

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __iter__(self):
        for i in [self.x, self.y]:
            yield i

    def __len__(self):
        return 2
    
    def __getitem__(self, idx):
        return (self.x, self.y)[idx]


# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


class IntersectCounter(solutions.ObjectCounter):
    def __init__(self, vertices: List[point], id, classes_names, scale: float = 1.0, ):
        self.vertices = [v * scale for v in vertices]
        # Init Object Counter
        super().__init__(
            view_img=True,
            reg_pts=self.vertices,
            classes_names=classes_names,
            draw_tracks=True,
            line_thickness=2,
        )
        self.id = int( id )

    @classmethod
    def init_centroid(cls, centroid, x_offset, y_offset, id, class_names, scale=1.0):
        p1, p2 = copy.deepcopy( centroid ), copy.deepcopy( centroid )
        vertices = [p1 + point( x_offset, y_offset ), p2 + point( -x_offset, -y_offset )]
        return cls( vertices, id, class_names, scale )

    @DeprecationWarning
    def check_intersect(self, path_vertices: List[point], object):
        A, B, C, D = self.vertices[0], self.vertices[1], path_vertices[0], path_vertices[1]
        if ccw( A, C, D ) != ccw( B, C, D ) and ccw( A, B, C ) != ccw( A, B, D ):
            self.count += 1
            self.counted_objects.append( object )
        return self.count

    def reset(self):
        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}

        # Tracks info
        self.track_history = defaultdict(list)

    def update_counters(self, frame, tracks, override_counter = None):
        if override_counter is not None:
            counters = override_counter # cannot access st.session state in webrtc
        else:
            counters = st.session_state.counters[self.tab_id]
        annontatedframe = frame
        if len( counters ) > 0:
            for p in counters:
                annontatedframe = p.counter.start_counting(annontatedframe, tracks)
        return annontatedframe




class st_IntersectCounter:
    def __init__(self, image, id, width, height, screen_width=1200):
        self.user_num_input = None
        self.user_cat_input = None
        self.to_filter_columns = None
        self.modification_container = None
        self.left = None
        self.right = None
        self.tab_id = id
        if 'counters_tables' not in st.session_state:
            st.session_state.counters_tables = {}
            self.counters_table = None
        else:
            self.counters_table = st.session_state.counters_tables.get(self.tab_id,None)
        if 'counters' not in st.session_state:
            st.session_state.counters = {}
        self.display_scale = width / screen_width *0.9
        self.counters_df_display = None
        self.counters_num = 0
        self.wrapper = st.expander( "**Setup Counter**" , expanded=True)
        self.option = 'Empty'
        self.counter_result_display = None
        screen_height = max(height // self.display_scale, 400)
        with self.wrapper:
            st.caption( "Draw lines on the below picture to set up counting function" )
            canvas_result = st_canvas(
                width=screen_width,
                height=screen_height,
                background_image=Image.fromarray( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ) ),
                stroke_width=1,
                drawing_mode='line', key="canvas"+str(self.tab_id)
            )

            if canvas_result.json_data is not None:
                if len( canvas_result.json_data['objects'] ) > 0:
                    self.canvas_result = canvas_result.json_data['objects']
                    self.counters_num = len( self.canvas_result )
                    self.format_counters_display()
                    all( self.generate_counters() )
                    st.markdown( '**Screenline Counters**' )
            if st.session_state.counters_tables.get(self.tab_id) is not None:
                self.counters_df_display = st.dataframe(
                    st.session_state.counters_tables.get(self.tab_id).style.format( precision=1 ), use_container_width=True, )
                        #st.markdown( '**Screenline Counters Result**' )

    def generate_counters(self):
        # if not st.session_state.counted:
        #     st.session_state.counters = []
        existed_counter = st.session_state.counters.get(self.tab_id, None)
        if existed_counter is None:
            st.session_state.counters[self.tab_id] = []
            existed_counter=st.session_state.counters.get(self.tab_id)
        for ind in range( len( existed_counter ), self.counters_num ):
            centroid = point( self.counters_table['left'][ind], self.counters_table['top'][ind] )
            xoffset = self.counters_table['x1'][ind]
            yoffset = self.counters_table['y1'][ind]
            counter = IntersectCounter.init_centroid( centroid, xoffset, yoffset, ind, st.session_state.class_names, self.display_scale )
            existed_counter.append( counter )
            self.sync_session_state()
            yield counter

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns
        https://github.com/tylerjrichards/st-filter-dataframe

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """

        df = df.copy()

        if 'to_filter_columns' not in st.session_state:
            st.session_state.to_filter_columns = None
        if 'user_cat_input' not in st.session_state:
            st.session_state.user_cat_input = None

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype( df[col] ):
                try:
                    df[col] = pd.to_datetime( df[col] )
                except Exception:
                    pass

            if is_datetime64_any_dtype( df[col] ):
                df[col] = df[col].dt.tz_localize( None )

        if self.modification_container is None:
            self.modification_container = st.container()

        with self.modification_container:
            if self.to_filter_columns is None:
                self.to_filter_columns = st.multiselect( "Filter dataframe on", df.columns )
            for column in self.to_filter_columns:
                if self.left is None:
                    self.left, self.right = st.columns( (1, 20) )
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype( df[column] ) or df[column].nunique() < 10:
                    if self.user_cat_input is None:
                        self.user_cat_input = self.right.multiselect(
                            f"Values for {column}",
                            df[column].unique(),
                            default=list( df[column].unique() ),
                        )
                    df = df[df[column].isin( self.user_cat_input )]
                elif is_numeric_dtype( df[column] ):
                    _min = float( df[column].min() )
                    _max = float( df[column].max() )
                    step = (_max - _min) / 100
                    if self.user_num_input is None:
                        self.user_num_input = user_num_input = self.right.slider(
                            f"Values for {column}",
                            min_value=_min,
                            max_value=_max,
                            value=(_min, _max),
                            step=step,
                            key='user_num_input'
                        )
                    df = df[df[column].between( *self.user_num_input )]
                elif is_datetime64_any_dtype( df[column] ):
                    user_date_input = self.right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len( user_date_input ) == 2:
                        user_date_input = tuple( map( pd.to_datetime, user_date_input ) )
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between( start_date, end_date )]
                else:
                    user_text_input = self.right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype( str ).str.contains( user_text_input )]

        return df

    def show_counter_results(self):
        if len( st.session_state.counters.get(self.tab_id, None) ) > 0:
            result_df = self.filter_dataframe(session_result.result_to_df([d.counted_objects for d in st.session_state.counters.get(self.tab_id, None)], counter_column=True) )
            result_df = result_df.style.background_gradient( axis=0, gmap=result_df['counter'], cmap='BuPu' )
            if self.counter_result_display is not None:
                self.counter_result_display.dataframe( result_df, use_container_width=True )
            else:
                self.counter_result_display = st.dataframe( result_df, use_container_width=True )

            self.counters_df_display.dataframe(
                self.format_counters_display().style.format( precision=1 ) )

    def format_counters_display(self):
        self.counters_table = pd.json_normalize( self.canvas_result )
        show_columns = ['type', 'left', 'top', 'x1', 'x2', 'y1', 'y2', 'width', 'height']
        if self.counters_table is not None:
            self.counters_table = self.counters_table[show_columns]
            # self.counters_table.style.format(precision=1)
        else:
            return None
        if len( st.session_state.counters.get(self.tab_id,[]) ) == len( self.counters_table ) and len( st.session_state.counters.get(self.tab_id,[]) ) > 0:
            self.counters_table['in_counts'] = [r.in_counts for r in st.session_state.counters[self.tab_id]]
            self.counters_table['out_counts'] = [r.out_counts for r in st.session_state.counters[self.tab_id]]
        self.sync_session_state()
        return self.counters_table

    def sync_session_state(self):
        st.session_state.counters_tables[self.tab_id] = self.counters_table

